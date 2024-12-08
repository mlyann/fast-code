import time
from datasets import load_dataset
import pandas as pd
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import numpy as np
from dotenv import load_dotenv
import nltk
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
from nltk import word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from torch.nn import functional as F


random.seed(721)
torch.manual_seed(721)
np.random.seed(721)


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
                output = tokenizer_large.decode(generated_tokens, skip_special_tokens=True)
                return output, len(generated_tokens)

            q_prob = approx_model_cache._prob_history[:, prefix_len + i - 1, j]
            p_prob = target_model_cache._prob_history[:, prefix_len + i - 1, j]
            acceptance_ratio = (p_prob / q_prob).clamp(max=1.0)

            if r < acceptance_ratio:
                generated_tokens.append(j.item())
                if verbose:
                    print(f"Token '{tokenizer_small.decode([j.item()])}' accepted with ratio {acceptance_ratio.item():.2f}")
                if tokenizer_large.decode([generated_tokens[-1]]) == '<|eot_id|>':
                    output = tokenizer_large.decode(generated_tokens, skip_special_tokens=True)
                    return output, len(generated_tokens)
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
                output = tokenizer_large.decode(generated_tokens, skip_special_tokens=True)
                return output, len(generated_tokens)
            input_ids = torch.cat([input_ids, t], dim=1)
        else:
            t = sample(target_model_cache._prob_history[:, -1, :])
            generated_tokens.append(t.item())
            if verbose:
                print(f"Sampling next token: '{tokenizer_large.decode([t.item()])}'")
            if tokenizer_large.decode([generated_tokens[-1]]) == '<|eot_id|>':
                output = tokenizer_large.decode(generated_tokens, skip_special_tokens=True)
                return output, len(generated_tokens)
            input_ids = torch.cat([input_ids, t], dim=1)
    output = tokenizer_large.decode(generated_tokens, skip_special_tokens=True)
    print(f"Output: \n{output}")
    return output, len(generated_tokens)



# Set up directories and device
results_dir = "results-fill-missing"
plots_dir = "plots"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load necessary libraries and data
load_dotenv()
nltk.download('stopwords')
nltk.download('punkt')

# Load GloVe embeddings
glove_model = api.load("glove-wiki-gigaword-100")

# Load dataset
dataset = load_dataset("openai_humaneval")

# Prepare data
num_samples = len(dataset["test"])  # Use all available samples
data = []
for i in range(num_samples):
    sample_data = {
        "task_id": dataset["test"][i]["task_id"],
        "prompt": dataset["test"][i]["prompt"],
        "canonical_solution": dataset["test"][i]["canonical_solution"],
        "test": dataset["test"][i]["test"],
        "entry_point": dataset["test"][i]["entry_point"]
    }
    data.append(sample_data)
original_df = pd.DataFrame(data)

# Function to mask continuous words
def mask_continuous_words(code, mask_ratio=0.1):
    words = code.split()
    total_words = len(words)
    num_to_mask = max(1, int(round(total_words * mask_ratio)))
    start_index = random.randint(0, total_words - num_to_mask)
    for i in range(start_index, start_index + num_to_mask):
        words[i] = ''
    return ' '.join(words)

# Apply masking
masked_prompts = [mask_continuous_words(code) for code in original_df['canonical_solution']]
original_df['masked'] = masked_prompts



# Define speculative_fix function
def speculative_fix(prompt, 
                    gamma=5, max_length=200, temperature=1.0, top_k=0, top_p=0.95, verbose=False):
    prompt = f"Please complete the following incomplete code to match the original solution. Do not add any extra code or function definitions. Only return the completed code, without any comments or explanations.\n\nHere is the code:\n\n{prompt}\n\nPlease provide the completed code:"
    output_text, num_tokens = speculative_decoding(prompt, gamma, max_length, temperature, top_k, top_p, verbose)
    return output_text, num_tokens

# Ensure both tokenizers are the same
assert tokenizer_small.get_vocab() == tokenizer_large.get_vocab(), "Tokenizers do not match!"

# Record time for speculative decoding
fixed_codes_speculative = []
speculative_times = []
speculative_tokens_counts = []

for code in tqdm(original_df['masked'], desc="Fixing code with Speculative Decoding"):
    start_time = time.time()
    fixed_code, num_tokens = speculative_fix(
        code,gamma=5, max_length=200, temperature=1.0, top_k=20, top_p=0.9, verbose=False
    )
    end_time = time.time()
    fixed_codes_speculative.append(fixed_code)
    speculative_times.append(end_time - start_time)
    speculative_tokens_counts.append(num_tokens)

# Add the results to the DataFrame
original_df["Fixed Code (Speculative Decoding)"] = fixed_codes_speculative

# Calculate average tokens per second
total_tokens = sum(speculative_tokens_counts)
total_time = sum(speculative_times)
average_tokens_per_second = total_tokens / total_time
print(f"Average tokens per second: {average_tokens_per_second:.2f} tokens/second")

# Evaluate models
def text_to_vector_list(text):
    if not isinstance(text, str):
        return []
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalpha() and word not in stopwords.words('english')]
    word_vectors = []
    for word in words:
        if word in glove_model:
            vec = glove_model[word]
            word_vectors.append(vec)
    return word_vectors

def calculate_max_cosine_similarity_word_by_word_with_sliding_window(df, column1, column2):
    vectors1 = df[column1].apply(text_to_vector_list)
    vectors2 = df[column2].apply(text_to_vector_list)
    
    similarities = []
    
    for vec_list1, vec_list2 in zip(vectors1, vectors2):
        if len(vec_list1) == 0 or len(vec_list2) == 0:
            similarities.append(0)
            continue
        
        shorter_vecs, longer_vecs = (vec_list1, vec_list2) if len(vec_list1) <= len(vec_list2) else (vec_list2, vec_list1)
        max_mean_similarity = -1
        
        for start in range(len(longer_vecs) - len(shorter_vecs) + 1):
            sims = [
                cosine_similarity([shorter_vecs[i]], [longer_vecs[start + i]])[0][0]
                for i in range(len(shorter_vecs))
            ]
            mean_similarity = np.mean(sims)
            if mean_similarity > max_mean_similarity:
                max_mean_similarity = mean_similarity
        
        similarities.append(max_mean_similarity)
    
    mean_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)
    return mean_similarity, std_similarity

# Evaluate speculative decoding results
avg_sim_spec, std_sim_spec = calculate_max_cosine_similarity_word_by_word_with_sliding_window(
    original_df, 'canonical_solution', "Fixed Code (Speculative Decoding)"
)
scores = [avg_sim_spec]
std_devs = [std_sim_spec]
labels = ['Speculative Decoding']

# Evaluate baseline (masked code)
avg_sim_base, std_sim_base = calculate_max_cosine_similarity_word_by_word_with_sliding_window(
    original_df, 'canonical_solution', 'masked'
)
scores.append(avg_sim_base)
std_devs.append(std_sim_base)
labels.append('Baseline')

# Plot the results
x = np.arange(len(labels))
fig, ax = plt.subplots()
ax.bar(x, scores, yerr=std_devs, capsize=5)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Cosine Similarity')
ax.set_title('Speculative Decoding vs. Baseline Performance')
plt.tight_layout()
plt.savefig("plots/speculative_decoding_vs_baseline_performance.png")

# Save the results
file_name = os.path.join(results_dir, "result-speculative_decoding.csv")
original_df.to_csv(file_name, index=False)

# 24.32 seconds VS 10.50 seconds
