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

# masking to  solutions
masked_prompts = [mask_continuous_words(code) for code in original_df['canonical_solution']]
masked_df = pd.DataFrame(masked_prompts, columns=['masked'])
final_df = pd.concat([original_df, masked_df], axis=1)

tokenizer = AutoTokenizer.from_pretrained("NTQAI/Nxcode-CQ-7B-orpo", use_fast=True)
model = AutoModelForCausalLM.from_pretrained("NTQAI/Nxcode-CQ-7B-orpo")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = model.to(device)

# fix calling GPT function
def fix(prompt, max_length=200):
    prompt = "Give me the fixed code without explanation: ```" + prompt + "```"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]
    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']
    outputs = model.generate(**inputs, max_new_tokens=max_length)
    generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    return generated_text

# fix code
masked_codes = final_df['masked']
results = []
for code in tqdm(masked_codes, desc="Fixing code"):
    result = fix(code)
    results.append(result)


results_df = pd.DataFrame(results, columns=['Fixed Code'])
final_df = pd.concat([final_df, results_df], axis=1)

final_df.to_csv('result-NTQAI-Nxcode-CQ-7B-orpo.csv', index=False)

