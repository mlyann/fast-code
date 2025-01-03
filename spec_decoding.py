

import torch
import numpy as np
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.bloom.modeling_bloom import BloomForCausalLM

# Set random seed for reproducibility
torch.manual_seed(721)
np.random.seed(721)

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

# Top-k and top-p filtering function
def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    if top_k > 0:
        filter = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < filter[:, [-1]]] = float('-inf')
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        filter = cumulative_probs > top_p
        filter[..., 1:] = filter[..., :-1].clone()
        filter[..., 0] = 0
        indices_to_remove = filter.scatter(1, sorted_indices, filter)
        logits[indices_to_remove] = float('-inf')
    return logits

def norm_logits(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    logits = logits / temperature
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=1)
    return probs

def sample(probs: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
    idx_next = torch.multinomial(probs, num_samples=num_samples)
    return idx_next

class KVCacheModel():
    def __init__(self, model: torch.nn.Module, temperature: float = 1, top_k: int = 0, top_p: float = 0):
        self._model = model
        self._past_key_values = None
        self._prob_history = None
        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p

    def _forward_with_kvcache(self, input_ids: torch.Tensor, use_debug: bool = True) -> torch.Tensor:
        if self._past_key_values is None:
            outputs = self._model(input_ids)
            self._prob_history = outputs.logits
            for i in range(self._prob_history.shape[-2]):   
                self._prob_history[:, i, :] = norm_logits(self._prob_history[:, i, :], self._temperature, self._top_k, self._top_p)
            self._past_key_values = outputs.past_key_values
            last_q = self._prob_history[:, -1, :]
        else:
            cached_len = self._past_key_values[0][0].shape[2]
            last_input_id = input_ids[:, cached_len:]
            if last_input_id.dim() == 1:
                last_input_id = torch.unsqueeze(last_input_id, 0)
            outputs = self._model(last_input_id, past_key_values=self._past_key_values, use_cache=True)
            not_cached_q = outputs.logits
            if not_cached_q.dim() == 2:
                not_cached_q = torch.unsqueeze(not_cached_q, 0)
            for i in range(not_cached_q.shape[-2]):   
                not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], self._temperature, self._top_k, self._top_p)    
            self._prob_history = torch.cat([self._prob_history, not_cached_q], dim=1)
            last_q = not_cached_q[:, -1, :]
            self._past_key_values = outputs.past_key_values
        return last_q

    def _generate_with_kvcache(self, prefix: torch.Tensor, gamma: int, use_debug=False) -> torch.Tensor:
        x = prefix
        for _ in range(gamma):
            q = self._forward_with_kvcache(x, use_debug)
            next_tok = sample(q)
            x = torch.cat((x, next_tok), dim=1)
        return x

    @torch.no_grad()
    def generate(self, input: torch.Tensor, gamma: int) -> torch.Tensor:
        output = self._generate_with_kvcache(input, gamma)
        return output
    
    @torch.no_grad()
    def rollback(self, end_pos: int):
        past_key_values_trimmed = []
        for kv in self._past_key_values:
            k, v = kv
            k = k[:, :, :end_pos, :]
            v = v[:, :, :end_pos, :]
            past_key_values_trimmed.append((k, v))
        self._past_key_values = past_key_values_trimmed
        self._prob_history = self._prob_history[:, :end_pos, :]

def speculative_decoding(prompt, gamma=5, max_length=200, temperature=1.0, top_k=0, top_p=0.95, verbose=False):
    input_ids = tokenizer_small.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    generated_tokens = []
    seq_len = input_ids.shape[1]
    T = seq_len + max_length

    approx_model_cache = KVCacheModel(model_small, temperature, top_k, top_p)
    target_model_cache = KVCacheModel(model_large, temperature, top_k, top_p)

    while input_ids.shape[1] < T:
        prefix_len = input_ids.shape[1]

        # Generate gamma tokens with the small model
        x = approx_model_cache.generate(input_ids, gamma)

        # Generate corresponding logits with the large model
        _ = target_model_cache.generate(x, gamma)

        n = prefix_len + gamma - 1

        # Check each token for acceptance
        for i in range(gamma):
            r = torch.rand(1, device=device)
            j = x[:, prefix_len + i]

            if j.item() == 0:
                return tokenizer_large.decode(generated_tokens, skip_special_tokens=True)

            q_prob = approx_model_cache._prob_history[:, prefix_len + i - 1, j]
            p_prob = target_model_cache._prob_history[:, prefix_len + i - 1, j]
            acceptance_ratio = (p_prob / q_prob).clamp(max=1.0)

            if r < acceptance_ratio:
                generated_tokens.append(j.item())
                if verbose:
                    print(f"Token '{tokenizer_small.decode([j.item()])}' accepted with ratio {acceptance_ratio.item():.2f}")
                if tokenizer_large.decode([generated_tokens[-1]]) == '<|eot_id|>':
                    return tokenizer_large.decode(generated_tokens, skip_special_tokens=True)
            else:
                if verbose:
                    print(f"Token '{tokenizer_small.decode([j.item()])}' rejected, will be replaced")
                n = prefix_len + i - 1
                break

        # Update input_ids based on acceptance
        input_ids = x[:, :n + 1]
        approx_model_cache.rollback(n + 1)
        target_model_cache.rollback(n + 1)

        if n < prefix_len + gamma - 1:
            delta_prob = (target_model_cache._prob_history[:, n, :] - approx_model_cache._prob_history[:, n, :]).clamp(min=0)
            delta_prob = delta_prob / delta_prob.sum(dim=-1, keepdim=True)
            t = sample(delta_prob)
            generated_tokens.append(t.item())
            if verbose:
                print(f"Token '{tokenizer_small.decode([x[0, n + 1].item()])}' rejected, replaced with '{tokenizer_large.decode([t.item()])}'")
            if tokenizer_large.decode([generated_tokens[-1]]) == '<|eot_id|>':
                return tokenizer_large.decode(generated_tokens, skip_special_tokens=True)
            input_ids = torch.cat([input_ids, t], dim=1)
        else:
            t = sample(target_model_cache._prob_history[:, -1, :])
            generated_tokens.append(t.item())
            if verbose:
                print(f"Sampling next token: '{tokenizer_large.decode([t.item()])}'")
            if tokenizer_large.decode([generated_tokens[-1]]) == '<|eot_id|>':
                return tokenizer_large.decode(generated_tokens, skip_special_tokens=True)
            input_ids = torch.cat([input_ids, t], dim=1)

    return tokenizer_large.decode(generated_tokens, skip_special_tokens=True)

# # Test example
# code = (
#     "for idx, elem in elem2 in enumerate(numbers): "
#     "if idx != idx2: distance = abs(elem - elem2) "
#     "if distance < threshold: return True "
#     "return False"
# )
# prompt = (
#     "Please complete the following incomplete code to match the original solution. "
#     "Do not add any extra code or function definitions. Only return the completed code, "
#     "without any comments or explanations.\n\nHere is the code:\n\n"
#     f"{code}\n\nPlease provide the completed code:"
# )

# output_text = speculative_decoding(prompt, gamma=5, max_length=200, temperature=1.0, top_k=0, top_p=0.95, verbose=False)
# print("\nFinal output text:")
# print(output_text)
