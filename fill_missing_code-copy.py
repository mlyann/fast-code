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

# Set random seeds
random.seed(721)
torch.manual_seed(721)
np.random.seed(721)

# Load large model and tokenizer
tokenizer_large = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B-Instruct", use_fast=True)
model_large = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B-Instruct", torch_dtype=torch.float16, device_map="auto"
)

# Set pad_token_id to eos_token_id to avoid warnings
tokenizer_large.pad_token_id = tokenizer_large.eos_token_id

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

# Knuth-Morris-Pratt (KMP) substring search
def kmp_search(context_tokens, target_tokens):
    """Find the first occurrence of target_tokens in context_tokens using KMP algorithm."""
    n, m = len(context_tokens), len(target_tokens)
    lps = [0] * m  # Longest prefix suffix array for target_tokens
    j = 0  # Index for target_tokens

    # Preprocess the target_tokens to create the LPS array
    i = 1
    while i < m:
        if target_tokens[i] == target_tokens[j]:
            j += 1
            lps[i] = j
            i += 1
        elif j > 0:
            j = lps[j - 1]
        else:
            lps[i] = 0
            i += 1

    # Search context_tokens for target_tokens using LPS array
    i, j = 0, 0  # i for context_tokens, j for target_tokens
    while i < n:
        if context_tokens[i] == target_tokens[j]:
            i += 1
            j += 1
            if j == m:
                return i - m  # Found match, return starting index
        elif j > 0:
            j = lps[j - 1]
        else:
            i += 1
    return -1  # No match found

# Updated copy decoding with copying mechanism
def copy_decoding_with_copying(prompt, gamma=5, max_length=200, verbose=False):
    input_ids = tokenizer_large.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    generated_tokens = []
    seq_len = input_ids.shape[1]
    T = seq_len + max_length

    target_model_cache = KVCacheModel(model_large, temperature=1.0, top_k=0, top_p=0.95)

    while input_ids.shape[1] < T:
        prefix_len = input_ids.shape[1]

        # Use copying mechanism to find the next gamma tokens
        context_tokens = input_ids[0].tolist()
        for i in range(gamma):
            # Check for matches in the last k tokens of the context
            k = 10  # Window size for matching
            match_idx = kmp_search(context_tokens[:prefix_len], context_tokens[prefix_len - k:])
            
            if match_idx != -1 and match_idx + k < len(context_tokens):
                # Found a match, copy the next token from the match position
                next_token = context_tokens[match_idx + k]
            else:
                # If no match or out of bounds, generate the next token using the large model
                q = target_model_cache._forward_with_kvcache(input_ids)
                next_token = sample(q).item()
            
            # Add the next token to the generated sequence
            generated_tokens.append(next_token)
            context_tokens.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]]).to(device)], dim=1)

            # Check for end of sequence
            if next_token == tokenizer_large.eos_token_id:
                return tokenizer_large.decode(generated_tokens, skip_special_tokens=True)

    return tokenizer_large.decode(generated_tokens, skip_special_tokens=True)

# Record time for copy decoding
fixed_codes_copy = []
copy_times = []

for code in tqdm(original_df['masked'], desc="Fixing code with copy Decoding"):
    start_time = time.time()
    fixed_code = copy_decoding_with_copying(
        code, gamma=5, max_length=200, verbose=False
    )
    end_time = time.time()
    fixed_codes_copy.append(fixed_code)
    copy_times.append(end_time - start_time)

# Add the results to the DataFrame
original_df["Fixed Code (copy Decoding)"] = fixed_codes_copy

# Record average time
average_copy_time = sum(copy_times) / len(copy_times)
print(f"Average time per copy decoding: {average_copy_time:.2f} seconds")
