from datasets import load_dataset
import pandas as pd
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")

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

# Apply masking to solutions
masked_prompts = [mask_continuous_words(code) for code in original_df['canonical_solution']]
original_df['masked'] = masked_prompts

# List of models to evaluate, adjust model paths if necessary
model_names = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "01-ai/Yi-Coder-9B-Chat",  # Adjust this path if necessary
    "microsoft/Phi-3-mini-128k-instruct",
    "google/codegemma-7b",  # Adjust this path if necessary
    "NTQAI/Nxcode-CQ-7B-orpo"  # Adjust this path if necessary
]

# Using GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Check for results directory
results_dir = "results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Function to fix code with a given model
def fix(prompt, model, tokenizer, max_length=200):
    prompt = "Here is some incomplete code:\n\n```" + prompt + "```\n\n I need the complete code, without any further explanation:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]
    outputs = model.generate(**inputs, max_new_tokens=max_length)
    generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    return generated_text

# Iterating over each model and fixing code
for model_name in model_names:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    
    fixed_codes = []
    for code in tqdm(original_df['masked'], desc=f"Fixing code with {model_name}"):
        fixed_code = fix(code, model, tokenizer)
        fixed_codes.append(fixed_code)
    
    # Adding fixed code to DataFrame
    model_column_name = f"Fixed Code ({model_name})"
    original_df[model_column_name] = fixed_codes

    # Save results to CSV in the results folder
    file_name = os.path.join(results_dir, f"result-{model_name.replace('/', '-')}.csv")  # Replace '/' in model name for file compatibility
    original_df.to_csv(file_name, index=False)

    print(f"Results saved to {file_name}")

print("All models processed and saved!")


