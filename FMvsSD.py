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
import time

# Set random seeds for reproducibility
random.seed(721)
torch.manual_seed(721)
np.random.seed(721)

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

hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
dataset = load_dataset("openai_humaneval")

# Load GloVe embeddings
glove_model = api.load("glove-wiki-gigaword-100")

# Prepare data
num_samples = 164
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

# List of models to evaluate
model_names = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "01-ai/Yi-Coder-9B-Chat",
    "microsoft/Phi-3-mini-128k-instruct",
    "google/codegemma-7b"
]
labels = ['Llama', 'Yi', 'Phi', 'Gemma']

# Function to fix code using the models
def fix(prompt, model, tokenizer, max_length=200):
    prompt = f"Please complete the following incomplete code to match the original solution. Do not add any extra code or function definitions. Only return the completed code, without any comments or explanations.\n\nHere is the code:\n\n```{prompt}```\n\nPlease provide the completed code:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]
    outputs = model.generate(**inputs, max_new_tokens=max_length)
    generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    return generated_text

# Process each model
for model_name in model_names:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

    fixed_codes = [fix(code, model, tokenizer) for code in tqdm(original_df['masked'], desc=f"Fixing code with {model_name}")]
    original_df[f"Fixed Code ({model_name})"] = fixed_codes

    file_name = os.path.join(results_dir, f"result-{model_name.replace('/', '-')}.csv")
    original_df.to_csv(file_name, index=False)

# -------------------- Speculative Decoding Addition -------------------- #

# Load small and large models for speculative decoding
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

# Function for speculative decoding
def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    if top_k > 0:
        filter = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < filter[:, [-1]]] = float('-inf')
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
        filter = cumulative_probs > top_p
        filter[..., 1:] = filter[..., :-1].clone()
        filter[..., 0] = 0
        indices_to_remove = filter.scatter(1, sorted_indices, filter)
        logits[indices_to_remove] = float('-inf')
    return logits

def norm_logits(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    assert logits.dim() == 2
    logits = logits / temperature
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    probs = torch.nn.functional.softmax(logits, dim=1)
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

    def _forward_with_kvcache(self, input_ids: torch.Tensor, use_debug: bool = False) -> torch.Tensor:
        if self._past_key_values is None:
            outputs = self._model(input_ids)
            self._prob_history = outputs.logits

            for i in range(self._prob_history.shape[-2]):
                self._prob_history[:, i, :] = norm_logits(self._prob_history[:, i, :], self._temperature, self._top_k, self._top_p)
            self._past_key_values = outputs.past_key_values
            last_q = self._prob_history[:, -1, :]
        else:
            cached_len = 0
            for kv in self._past_key_values:
                k, v = kv
                cached_len = k.shape[2]

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
        assert self._past_key_values
        for kv in self._past_key_values:
            k, v = kv
            # k, v (batch, head, seq, hidden_dim)
            k = k[:, :, :end_pos, :]
            v = v[:, :, :end_pos, :]
            kv_trimmed = (k, v)
            past_key_values_trimmed.append(kv_trimmed)

        self._past_key_values = past_key_values_trimmed
        self._prob_history = self._prob_history[:, :end_pos, :]

def speculative_decoding(prompt, gamma=5, max_length=200, temperature=1.0, top_k=0, top_p=0.95):
    input_ids = tokenizer_small.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    attention_mask = torch.ones_like(input_ids)

    generated_tokens = []
    total_generated = 0
    seq_len = input_ids.shape[1]
    T = seq_len + max_length

    approx_model_cache = KVCacheModel(model_small, temperature, top_k, top_p)
    target_model_cache = KVCacheModel(model_large, temperature, top_k, top_p)

    while input_ids.shape[1] < T:
        prefix_len = input_ids.shape[1]
        x = approx_model_cache.generate(input_ids, gamma)
        _ = target_model_cache.generate(x, gamma)
        n = prefix_len + gamma - 1

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
            else:
                n = prefix_len + i - 1
                break

        input_ids = x[:, :n + 1]
        approx_model_cache.rollback(n + 1)
        target_model_cache.rollback(n + 1)

        if n < prefix_len + gamma - 1:
            delta_prob = (target_model_cache._prob_history[:, n, :] - approx_model_cache._prob_history[:, n, :]).clamp(min=0)
            delta_prob = delta_prob / delta_prob.sum(dim=-1, keepdim=True)
            t = sample(delta_prob)
            generated_tokens.append(t.item())
            if t.item() == 0:
                return tokenizer_large.decode(generated_tokens, skip_special_tokens=True)
            input_ids = torch.cat([input_ids, t], dim=1)
            total_generated += 1
            continue
        else:
            t = sample(target_model_cache._prob_history[:, -1, :])
            generated_tokens.append(t.item())
            if t.item() == 0:
                return tokenizer_large.decode(generated_tokens, skip_special_tokens=True)
            input_ids = torch.cat([input_ids, t], dim=1)
            total_generated += gamma + 1

    output_text = tokenizer_large.decode(generated_tokens, skip_special_tokens=True)
    return output_text

# Timer function for speculative decoding
def timed_speculative_decoding(prompt, **kwargs):
    start_time = time.time()
    output_text = speculative_decoding(prompt, **kwargs)
    elapsed_time = time.time() - start_time
    return output_text, elapsed_time

# Record times and outputs for speculative decoding
speculative_times = []
speculative_outputs = []

# Prepare prompts for speculative decoding
speculative_prompts = []
for code in original_df['masked']:
    prompt = f"Please complete the following incomplete code to match the original solution. Do not add any extra code or function definitions. Only return the completed code, without any comments or explanations.\n\nHere is the code:\n\n```{code}```\n\nPlease provide the completed code:"
    speculative_prompts.append(prompt)

# Run speculative decoding and record outputs and times
for prompt in tqdm(speculative_prompts, desc="Speculative Decoding"):
    output_text, elapsed_time = timed_speculative_decoding(prompt, gamma=5, max_length=200, temperature=1.0, top_k=0, top_p=0.9)
    speculative_outputs.append(output_text)
    speculative_times.append(elapsed_time)

# Add speculative decoding outputs to DataFrame
original_df['Fixed Code (Speculative)'] = speculative_outputs

# Save results to CSV
file_name = os.path.join(results_dir, "result-Speculative_Decoding.csv")
original_df.to_csv(file_name, index=False)

# -------------------- Evaluation and Plotting -------------------- #

# Function to convert text to list of vectors
def text_to_vector_list(text):
    if not isinstance(text, str):
        return []
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalpha() and word not in stopwords.words('english')]
    word_vectors = []
    for word in words:
        if word in glove_model:
            vec = glove_model[word]
            if vec.size > 0:
                word_vectors.append(vec)
            else:
                print(f"Warning: The vector for word '{word}' is empty.")
        else:
            # print(f"Warning: Word '{word}' not found in the GloVe model.")
            pass
    return word_vectors

# Function to calculate max cosine similarity with sliding window
def calculate_max_cosine_similarity_word_by_word_with_sliding_window(df, column1, column2):
    vectors1 = df[column1].apply(text_to_vector_list)
    vectors2 = df[column2].apply(text_to_vector_list)

    similarities = []

    for vec_list1, vec_list2 in zip(vectors1, vectors2):
        if len(vec_list1) <= len(vec_list2):
            shorter_vecs = vec_list1
            longer_vecs = vec_list2
        else:
            shorter_vecs = vec_list2
            longer_vecs = vec_list1

        if not shorter_vecs or not longer_vecs:
            similarities.append(0)
            continue

        max_mean_similarity = -1

        for start in range(len(longer_vecs) - len(shorter_vecs) + 1):
            sims = []
            for i in range(len(shorter_vecs)):
                vec1 = shorter_vecs[i]
                vec2 = longer_vecs[start + i]
                if vec1.size == 0 or vec2.size == 0:
                    # print(f"Warning: Encountered empty vector at position {i}.")
                    continue
                similarity = cosine_similarity([vec1], [vec2])[0][0]
                sims.append(similarity)
            if sims:
                mean_similarity = np.mean(sims)
                if mean_similarity > max_mean_similarity:
                    max_mean_similarity = mean_similarity

        similarities.append(max_mean_similarity)

    mean_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)
    return mean_similarity, std_similarity

# Load results
results = [pd.read_csv(os.path.join(results_dir, f"result-{model_name.replace('/', '-')}.csv")) for model_name in model_names]
results.append(pd.read_csv(os.path.join(results_dir, "result-Speculative_Decoding.csv")))

# Update labels to include speculative decoding
labels = ['Llama', 'Yi', 'Phi', 'Gemma', 'Speculative']

# Evaluate models
scores, std_devs = [], []
for df, label in zip(results, labels):
    avg_sim, std_sim = calculate_max_cosine_similarity_word_by_word_with_sliding_window(df, 'canonical_solution', f"Fixed Code ({label})")
    scores.append(avg_sim)
    std_devs.append(std_sim)

# Evaluate baseline ('masked' column)
df_baseline = results[0]  # The 'masked' column is the same across all dataframes
avg_sim_base, std_sim_base = calculate_max_cosine_similarity_word_by_word_with_sliding_window(df_baseline, 'canonical_solution', 'masked')

# Append baseline to scores and labels
scores.append(avg_sim_base)
std_devs.append(std_sim_base)
labels.append('Baseline')

# Plot model performance including baseline
x = np.arange(len(labels))
fig, ax = plt.subplots()
ax.bar(x, scores, yerr=std_devs, capsize=5)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)
ax.set_ylabel('Cosine Similarity')
ax.set_title('Fill Missing Code Model Performance Evaluation - Including Speculative Decoding')
plt.tight_layout()
plt.savefig("plots/fill-missing-code-model_performance_evaluation-with-speculative-decoding.png")

# Record times for original models (assuming you have recorded them)
# For demonstration, I'll initialize empty lists
times_original_models = [[] for _ in model_names]  # Replace with your actual timing lists

# Record times for speculative decoding
times_speculative = speculative_times

# Ensure all timing lists are of length 164
# times_original_models = [your_times_list_for_each_model]  # Replace with actual data

# For plotting, we can calculate average times
average_times = []
average_times_std = []
for times in times_original_models:
    average_times.append(np.mean(times))
    average_times_std.append(np.std(times))

# Append speculative decoding times
average_times.append(np.mean(times_speculative))
average_times_std.append(np.std(times_speculative))

# Update labels to include speculative decoding
labels_time = ['Llama', 'Yi', 'Phi', 'Gemma', 'Speculative']

# Plot timing comparison
x = np.arange(len(labels_time))
fig, ax = plt.subplots()
ax.bar(x, average_times, yerr=average_times_std, capsize=5)
ax.set_xticks(x)
ax.set_xticklabels(labels_time, rotation=45)
ax.set_ylabel('Average Time (s)')
ax.set_title('Average Generation Time Comparison - Including Speculative Decoding')
plt.tight_layout()
plt.savefig('plots/generation_time_comparison_with_speculative_decoding.png')

# Optionally, you can also plot per-sample times if you have the data
