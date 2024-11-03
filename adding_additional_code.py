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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk import word_tokenize
from nltk.corpus import stopwords
import re
import matplotlib.pyplot as plt

random.seed(721)
torch.manual_seed(721)
np.random.seed(721)

# Process each model
results_dir = "results-adding-noise"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Load necessary libraries and data
load_dotenv()
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')


hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
dataset = load_dataset("openai_humaneval")
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

# Masking function
def adding_continuous_words(code, add_ratio=0.1):
    words = code.split()
    total_words = len(words)
    num_to_add = max(1, int(round(total_words * add_ratio)))
    start_index = random.randint(0, total_words - 1)
    for i in range(start_index, start_index + num_to_add):
        words.insert(i, 'new_word')
    return ' '.join(words)

adding_prompts = [adding_continuous_words(code) for code in original_df['canonical_solution']]
original_df['Noisy_Code'] = adding_prompts


# List of models to evaluate
model_names = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "01-ai/Yi-Coder-9B-Chat",
    "microsoft/Phi-3-mini-128k-instruct",
    "google/codegemma-7b"
]

# Setup GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to fix code
def fix(prompt, model, tokenizer, max_length=200):
    prompt = f"Here is some code with additional noisy characters inserting into:\n\n```{prompt}```\n\nGive me the complete code, without any further explanation:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]
    outputs = model.generate(**inputs, max_new_tokens=max_length)
    generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    return generated_text



for model_name in model_names:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    fixed_codes = [fix(code, model, tokenizer) for code in tqdm(original_df['Noisy_Code'], desc=f"Fixing code with {model_name}")]
    original_df[f"Fixed Code ({model_name})"] = fixed_codes

    file_name = os.path.join(results_dir, f"result-{model_name.replace('/', '-')}.csv")
    original_df.to_csv(file_name, index=False)


def text_to_vector(text):
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalpha() and word not in stopwords.words('english')]
    word_vectors = [glove_model[word] for word in words if word in glove_model]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(glove_model.vector_size)

def calculate_max_cosine_similarity_with_sliding_window(df, column1, column2):
    vectors1 = df[column1].apply(text_to_vector)
    vectors2 = df[column2].apply(text_to_vector)
    shorter_vectors, longer_vectors = vectors1, vectors2
    
    shorter_matrix = np.stack(shorter_vectors.values)
    
    similarities = []
    
    for start in range(len(longer_vectors) - len(shorter_matrix) + 1):
        window_matrix = np.stack(longer_vectors[start:start + len(shorter_matrix)].values)
        window_similarities = cosine_similarity(shorter_matrix, window_matrix).diagonal()
        mean_similarity = np.mean(window_similarities)
        
        similarities.append(mean_similarity)
    max_similarity = max(similarities)
    avg_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)
    return avg_similarity, std_similarity


# Load results and compute similarity
results = [pd.read_csv(os.path.join(results_dir, f"result-{model_name.replace('/', '-')}.csv")) for model_name in model_names]
labels = ['Llama', 'Yi', 'Phi', 'Gemma']

scores, std_devs = [], []
for df, model_name in zip(results, model_names):
    avg_sim, std_sim = calculate_max_cosine_similarity_with_sliding_window(df, 'canonical_solution', f"Fixed Code ({model_name})")
    scores.append(avg_sim)
    std_devs.append(std_sim)

x = np.arange(len(labels))
fig, ax = plt.subplots()
ax.bar(x, scores, yerr=std_devs, capsize=5)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Cosine Similarity')
ax.set_title('Adding Noise Performance Evaluation')

# Save the figure as a PNG file
plt.savefig("adding_noise_model_performance_evaluation.png")



scores_base, std_devs_base = [], []
for df, model_name in zip(results, model_names):
    avg_sim_base, std_sim_base = calculate_max_cosine_similarity_with_sliding_window(df, 'canonical_solution', "Noisy_Code")
    scores_base.append(avg_sim_base)
    std_devs_base.append(std_sim_base)

x = np.arange(len(labels))
fig, ax = plt.subplots()
ax.bar(x, scores_base, yerr=std_devs_base, capsize=5)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Cosine Similarity')
ax.set_title('Adding Noise Baseline')

# Save the figure as a PNG file
plt.savefig("adding_noise_baseline.png")
