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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)




tokenizer_small = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", use_fast=True)
model_small = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.float16, device_map="auto"
)

tokenizer_large = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B-Instruct", use_fast=True)
model_large = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B-Instruct", torch_dtype=torch.float16, device_map="auto"
)



# Utils functions
def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    if top_k > 0:
        filter_val = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < filter_val[:, [-1]]] = float('-inf')
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        filter_val = cumulative_probs > top_p
        filter_val[..., 1:] = filter_val[..., :-1].clone()
        filter_val[..., 0] = 0
        logits[filter_val.scatter(1, sorted_indices, filter_val)] = float('-inf')
    return logits


def norm_logits(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    logits = logits / temperature
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    return F.softmax(logits, dim=1)


def sample(probs: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
    return torch.multinomial(probs, num_samples=num_samples)


# KVCacheModel class
class KVCacheModel:
    def __init__(self, model: torch.nn.Module, temperature: float = 1, top_k: int = 0, top_p: float = 0):
        self._model = model
        self._past_key_values = None
        self._prob_history = None
        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p

    def _forward_with_kvcache(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self._past_key_values is None:
            outputs = self._model(input_ids)
            self._prob_history = outputs.logits
            for i in range(self._prob_history.shape[-2]):
                self._prob_history[:, i, :] = norm_logits(self._prob_history[:, i, :], self._temperature, self._top_k, self._top_p)
            self._past_key_values = outputs.past_key_values
            return self._prob_history[:, -1, :]
        else:
            cached_len = self._past_key_values[0][0].shape[2]
            last_input_id = input_ids[:, cached_len:]
            outputs = self._model(last_input_id, past_key_values=self._past_key_values, use_cache=True)
            not_cached_q = outputs.logits
            for i in range(not_cached_q.shape[-2]):
                not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], self._temperature, self._top_k, self._top_p)
            self._prob_history = torch.cat([self._prob_history, not_cached_q], dim=1)
            self._past_key_values = outputs.past_key_values
            return not_cached_q[:, -1, :]

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, gamma: int) -> torch.Tensor:
        output = input_ids
        for _ in range(gamma):
            next_prob = self._forward_with_kvcache(output)
            next_token = sample(next_prob)
            output = torch.cat((output, next_token), dim=1)
        return output

    @torch.no_grad()
    def rollback(self, end_pos: int):
        self._past_key_values = [
            (k[:, :, :end_pos, :], v[:, :, :end_pos, :])
            for k, v in self._past_key_values
        ]
        self._prob_history = self._prob_history[:, :end_pos, :]


# Speculative Decoding
@torch.no_grad()
def speculative_sampling_v2(prefix, approx_model, target_model, max_len, gamma, temperature, top_k, top_p):
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    prefix = torch.clone(prefix)
    approx_model_cache = KVCacheModel(approx_model, temperature, top_k, top_p)
    target_model_cache = KVCacheModel(target_model, temperature, top_k, top_p)
    while prefix.shape[1] < T:
        x = approx_model_cache.generate(prefix, gamma)
        target_model_cache.generate(x, gamma)
    return prefix


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
dataset = load_dataset("openai_humaneval")
num_samples = len(dataset["test"])
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
masked_prompts = [mask_continuous_words(code) for code in original_df['canonical_solution']]
original_df['masked'] = masked_prompts


# Speculative Fix v2
def speculative_fix_v2(prompt, 
                       gamma=5, max_length=200, temperature=1.0, top_k=50, top_p=0.9):
    prompt = f"Please complete the following incomplete code to match the original solution. Do not add any extra code or function definitions. Only return the completed code, without any comments or explanations.\n\nHere is the code:\n\n{prompt}\n\nPlease provide the completed code:"
    input_ids = tokenizer_small.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    output_tokens = speculative_sampling_v2(
        prefix=input_ids,
        approx_model=model_small,
        target_model=model_large,
        max_len=max_length,
        gamma=gamma,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    return tokenizer_large.decode(output_tokens[0], skip_special_tokens=True)


# Speculative Decoding v2 Execution
fixed_codes_speculative_v2 = []
speculative_v2_times = []

for code in tqdm(original_df['masked'], desc="Fixing code with Speculative Decoding v2"):
    start_time = time.time()
    fixed_code = speculative_fix_v2(code, gamma=5, max_length=200, temperature=1.0, top_k=50, top_p=0.9)
    end_time = time.time()
    fixed_codes_speculative_v2.append(fixed_code)
    speculative_v2_times.append(end_time - start_time)

# Add v2 results to DataFrame
original_df["Fixed Code (Speculative Decoding v2)"] = fixed_codes_speculative_v2

# Save results
average_speculative_v2_time = sum(speculative_v2_times) / len(speculative_v2_times)
print(f"Average time per speculative decoding v2: {average_speculative_v2_time:.2f} seconds")
results_dir = "results-fill-missing"
os.makedirs(results_dir, exist_ok=True)
original_df.to_csv(os.path.join(results_dir, "result-speculative_decoding_v2.csv"), index=False)
