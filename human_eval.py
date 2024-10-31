# Import necessary libraries
from datasets import load_dataset
import pandas as pd
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load dataset
dataset = load_dataset("openai_humaneval")
num_samples = 164

# Create original DataFrame
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

# Masking function
def mask_continuous_words(code, mask_ratio=0.1):
    words = code.split()
    total_words = len(words)
    num_to_mask = max(1, int(round(total_words * mask_ratio)))
    start_index = random.randint(0, total_words - num_to_mask)
    for i in range(start_index, start_index + num_to_mask):
        words[i] = ''
    return ' '.join(words)

# Apply masking to canonical solutions
masked_prompts = [mask_continuous_words(code) for code in original_df['canonical_solution']]
masked_df = pd.DataFrame(masked_prompts, columns=['masked'])
final_df = pd.concat([original_df, masked_df], axis=1)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", use_fast=True)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = model.to(device)

# Define fix function
def fix(prompt, max_length=200):
    prompt = "Give me the fixed code without explanation: ```" + prompt + "```"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]
    outputs = model.generate(**inputs, max_new_tokens=max_length)
    generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    return generated_text

# Fix code in masked prompts
masked_codes = final_df['masked']
results = []
for code in tqdm(masked_codes, desc="Fixing code"):
    result = fix(code)
    results.append(result)

# Combine results into DataFrame
results_df = pd.DataFrame(results, columns=['Fixed Code'])
final_df = pd.concat([final_df, results_df], axis=1)

# Save final DataFrame to CSV
final_df.to_csv('result.csv', index=False)

