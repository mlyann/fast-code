from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import torch
from torch.nn import functional as F
import time
import torch
from typing import List, Optional
from transformers.models.bloom.modeling_bloom import BloomForCausalLM


# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load small and large models
tokenizer_small = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", use_fast=True)
model_small = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.float16, device_map="auto"
)

tokenizer_large = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B-Instruct", use_fast=True)
model_large = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B-Instruct", torch_dtype=torch.float16, device_map="auto"
)

# Set pad_token_id to eos_token_id to avoid warnings
tokenizer_small.pad_token_id = tokenizer_small.eos_token_id
tokenizer_large.pad_token_id = tokenizer_large.eos_token_id

# Ensure both tokenizers are the same
assert tokenizer_small.get_vocab() == tokenizer_large.get_vocab(), "Tokenizers do not match!"



# copy from https://github.com/LeeSinLiang/microGPT/blob/ed40cf9780dbeb180adfe94c227d4aa97e69250e/gpt.py
def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    """

    Args:
        logits (torch.Tensorpe_): 2D tensor with shape (batch, vocab)
        top_k (int, optional): top_k. Defaults to 0.
        top_p (float, optional): top_p. Defaults to 0.0.

    Returns:
        torch.Tensor: a renormalized logits
    """
    if top_k > 0:
        filter = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < filter[:, [-1]]] = float('-inf')
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)
        filter = cumulative_probs > top_p
        filter[..., 1:] = filter[..., :-1].clone()
        filter[..., 0] = 0
        indices_to_remove = filter.scatter(1, sorted_indices, filter)
        logits[indices_to_remove] = float('-inf')
    return logits


def norm_logits(logits : torch.Tensor, temperature : float, top_k : float, top_p : float) -> torch.Tensor:
    """

    Args:
        logits (torch.Tensor): shape (1, vocab)
        temperature (float): temperature
        top_k (float): top_k
        top_p (float): top_p

    Returns:
        torch.Tensor: next token with shape as (batch,  1)
    """
    assert logits.dim() == 2
    logits = logits / temperature
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=1)
    return probs


def sample(probs: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
    """
    Samples indices from a probability distribution tensor.

    Args:
        probs (torch.Tensor): A tensor containing probability values, where each 
            element signifies the probability of picking the corresponding index.
        num_samples (int, optional): The number of indices to sample. Defaults to 1.

    Returns:
        torch.Tensor: A tensor containing the sampled indices based on the probability distribution.

    Raises:
        RuntimeError: If the sampled index is 0, which may indicate improper sampling or a model issue.
    """
    idx_next = torch.multinomial(probs, num_samples=num_samples)
    if idx_next.item() == 0:
        raise RuntimeError("Sampled index is zero, it should stop.")
    return idx_next

def max_fn(x):
    """
    Compute the normalized positive elements of the input tensor x.

    The function performs the following steps:
    1. Replace negative elements of x with zero, effectively applying a ReLU operation.
    2. Compute the sum of the positive elements along the specified dimension (dim=1).
    3. Normalize the positive elements of x by dividing them by their respective sums.

    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: Normalized tensor containing only the positive elements of x.
    """
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=1, keepdim=True)
    return x_max / x_max_sum

def _debug_show_kvcache(past_key_values):
    """
    Print the shapes of keys and values from a key-value cache, for debugging purposes.

    If the past_key_values is None, the function returns immediately without doing anything.
    Otherwise, iterates over the past_key_values and retrieves the first element (pair of key and value),
    printing their shapes.

    Args:
        past_key_values (Iterable[Tuple[Tensor, Tensor]]): An iterable containing tuples of keys and values tensors.
    """
    if past_key_values is None:
        return
    for elem in past_key_values:
        k, v = elem
        print(f"kv cache: k shape {k.shape}, v shape {v.shape}")
        break

class KVCacheModel():
    def __init__(self, model: torch.nn.Module, temperature: float = 1, top_k: int = 0, top_p: float = 0) -> None:
        """
        Initializes a KVCacheModel instance.

        Args:
            model (torch.nn.Module): The model to be used for generating logits and managing key-value caches.
            temperature (float, optional): The temperature value for adjusting the distribution of logits. Defaults to 1.
            top_k (int, optional): The number of highest probability logits to keep. Defaults to 0.
            top_p (float, optional): The cumulative probability threshold for probability mass of logits. Defaults to 0.
        """
        self._model = model
        self._past_key_values = None
        self._prob_history = None

        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p

    def _forward_with_kvcache(self, input_ids: torch.Tensor, use_debug: bool = True) -> torch.Tensor:
        """
        Forward pass with key-value caching to efficiently generate logits.

        Args:
            input_ids (torch.Tensor): The input tokens for the model.
            use_debug (bool, optional): If set to True, prints debugging information. Defaults to True.

        Returns:
            torch.Tensor: Last token's logits after normalizing with the temperature and filtering with top_k and top_p.
        """
        if self._past_key_values is None:
            # First forward pass, where no past key-values are set (prefill)
            assert self._prob_history is None, f"{self._prob_history.shape}"
            outputs = self._model(input_ids)
            self._prob_history = outputs.logits

            # Normalize and filter logits for the entire prompt
            for i in range(self._prob_history.shape[-2]):   
                self._prob_history[:, i, :] = norm_logits(self._prob_history[:, i, :], self._temperature, self._top_k, self._top_p)
            self._past_key_values = outputs.past_key_values
            last_q = self._prob_history[:, -1, :]
        else:
            cached_len = 0
            for kv in self._past_key_values:
                k, v = kv
                cached_len = k.shape[2]
                
            # Get new input IDs that haven't been processed
            last_input_id = input_ids[:, cached_len:]
            if last_input_id.dim() == 1:
                last_input_id = torch.unsqueeze(last_input_id, 0)
            
            if use_debug:
                print(f"last_input_id shape {last_input_id.shape}")
                _debug_show_kvcache(self._past_key_values)
            
            # Forward pass with cached key-values
            outputs = self._model(last_input_id, past_key_values=self._past_key_values, use_cache=True)
            
            not_cached_q = outputs.logits
            if not_cached_q.dim() == 2:
                not_cached_q = torch.unsqueeze(not_cached_q, 0)
                
            for i in range(not_cached_q.shape[-2]):   
                not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], self._temperature, self._top_k, self._top_p)    
                
            # Update probability with new logits
            self._prob_history = torch.cat([self._prob_history, not_cached_q], dim=1)
            
            last_q = not_cached_q[:, -1, :]
            self._past_key_values = outputs.past_key_values
        
        return last_q

    def _generate_with_kvcache(self, prefix : torch.Tensor, 
                                    gamma : int, 
                                    use_debug = False) -> torch.Tensor:
        """ forward the model gamma times

        Args:
            prefix (torch.Tensor): the prefix
            gamma (int): how many times approx guesses

        Returns:
            Torch.Tensor: prefix+generated tokens
        """
        x = prefix

        for _ in range(gamma):
            q = self._forward_with_kvcache(x, use_debug)
            next_tok = sample(q)
            x = torch.cat((x, next_tok), dim=1)
        return x

    @torch.no_grad()
    def generate(self, input : torch.Tensor, gamma : int) -> torch.Tensor:
        output = self._generate_with_kvcache(input, gamma)
        return output
    
    @torch.no_grad()
    def rollback(self, end_pos : int):
        past_key_values_trimmed = []
        assert self._past_key_values
        for kv in self._past_key_values:
            k, v = kv
            # NOTE() the indexing is specific for bloom. This won't work for other models
            # For example llama k, v should be (batch, num_head, seq_len, hidden_dim)
            
            # Bloom is special one
            if isinstance(self._model, BloomForCausalLM):
                # k (batch * head, hidden_dim, seq); v (batch * head, seq, hidden_dim)
                k = k[:, :, :end_pos]
                v = v[:, :end_pos, :]
                kv_trimmed = (k, v)
                past_key_values_trimmed.append(kv_trimmed)
            else:
                # k, v (batch, head, seq, hidden_dim)
                k = k[:, :, :end_pos, :]
                v = v[:, :, :end_pos, :]
                kv_trimmed = (k, v)
                past_key_values_trimmed.append(kv_trimmed)
        
        self._past_key_values = past_key_values_trimmed
        self._prob_history = self._prob_history[:, :end_pos, :]

    
def speculative_decoding(prompt, gamma=5, max_length=200, temperature=1.0, top_k=0, top_p=0.9, verbose=False):
    input_ids = tokenizer_small.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    attention_mask = torch.ones_like(input_ids)

    generated_tokens = []
    total_generated = 0
    seq_len = input_ids.shape[1]
    T = seq_len + max_length

    # Initialize KVCacheModel instances
    approx_model_cache = KVCacheModel(model_small, temperature, top_k, top_p)
    target_model_cache = KVCacheModel(model_large, temperature, top_k, top_p)

    while input_ids.shape[1] < T:
        prefix_len = input_ids.shape[1]

        # Step 1: Generate gamma tokens with small model
        x = approx_model_cache.generate(input_ids, gamma)

        # Step 2: Generate corresponding logits with the large model
        _ = target_model_cache.generate(x, 1)

        n = prefix_len + gamma - 1

        # Step 3: Check each token for acceptance
        for i in range(gamma):
            r = torch.rand(1, device=device)
            j = x[:, prefix_len + i]

            # Check if generated token is 0
            if j.item() == 0:
                print("Terminating generation due to 0 token.")
                return tokenizer_large.decode(generated_tokens, skip_special_tokens=True)

            # Compute acceptance probability
            q_prob = approx_model_cache._prob_history[:, prefix_len + i - 1, j]
            p_prob = target_model_cache._prob_history[:, prefix_len + i - 1, j]
            acceptance_ratio = (p_prob / q_prob).clamp(max=1.0)

            if r < acceptance_ratio:
                # Accept the token
                if verbose:
                    print(f"Token '{tokenizer_small.decode([j.item()])}' accepted with ratio {acceptance_ratio.item():.2f}")
                generated_tokens.append(j.item())
            else:
                # Reject and sample from the large model
                n = prefix_len + i - 1
                break

        # Update input_ids based on acceptance
        input_ids = x[:, :n + 1]
        approx_model_cache.rollback(n + 1)
        target_model_cache.rollback(n + 1)

        if n < prefix_len + gamma - 1:
            # Token was rejected, sample from the large model
            delta_prob = (target_model_cache._prob_history[:, n, :] - approx_model_cache._prob_history[:, n, :]).clamp(min=0)
            delta_prob = delta_prob / delta_prob.sum(dim=-1, keepdim=True)
            t = sample(delta_prob)  # t has shape [1, 1]
            generated_tokens.append(t.item())
            if verbose:
                print(f"Token '{tokenizer_small.decode([x[0, n + 1].item()])}' rejected, replaced with '{tokenizer_large.decode([t.item()])}'")
            # Check if sampled token is 0
            if t.item() == 0:
                print("Terminating generation due to 0 token.")
                return tokenizer_large.decode(generated_tokens, skip_special_tokens=True)
            # Update input_ids with the replacement token
            input_ids = torch.cat([input_ids, t], dim=1)
            total_generated += 1
            continue  # Restart generation after rejection

        else:
            # All tokens were accepted; sample the next token from the large model
            t = sample(target_model_cache._prob_history[:, -1, :])  # t has shape [1, 1]
            generated_tokens.append(t.item())
            if verbose:
                print(f"Sampling next token: '{tokenizer_large.decode([t.item()])}'")
            # Check if sampled token is 0
            if t.item() == 0:
                print("Terminating generation due to 0 token.")
                return tokenizer_large.decode(generated_tokens, skip_special_tokens=True)
            # Update input_ids with the new token
            input_ids = torch.cat([input_ids, t], dim=1)
            total_generated += gamma + 1

    # Decode and return the generated text
    output_text = tokenizer_large.decode(generated_tokens, skip_special_tokens=True)
    return output_text

# Test example
code = (
    "for idx, elem in elem2 in enumerate(numbers): "
    "if idx != idx2: distance = abs(elem - elem2) "
    "if distance < threshold: return True "
    "return False"
)
prompt = (
    "Please complete the following incomplete code to match the original solution. "
    "Do not add any extra code or function definitions. Only return the completed code, "
    "without any comments or explanations.\n\nHere is the code:\n\n"
    f"{code}\n\nPlease provide the completed code:"
)

output_text = speculative_decoding(prompt, gamma=5, max_length=200, temperature=1.0, top_k=0, top_p=0.9, verbose=True)
print("\nFinal output text:")
print(output_text)
