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

# Set random seeds for reproducibility
random.seed(721)
torch.manual_seed(721)
np.random.seed(721)

# Set up directories and device
results_dir = "results-adding-noise"
plots_dir = "plots"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

# Function to add random words at random positions
def add_random_words(code, add_ratio=0.1):
    words = code.split()
    total_words = len(words)
    num_to_add = max(1, int(round(total_words * add_ratio)))
    for _ in range(num_to_add):
        random_word = random.choice(list(glove_model.key_to_index.keys()))
        insert_position = random.randint(0, len(words))
        words.insert(insert_position, random_word)
    return ' '.join(words)

# Apply the function to add noise to the code
noisy_codes = [add_random_words(code) for code in original_df['canonical_solution']]
original_df['Noisy_Code'] = noisy_codes

# List of models to evaluate
model_names = [
    "meta-llama/Llama-3.1-70B-Instruct",
    #"Qwen/Qwen2-72B-Instruct"
]
labels = ['Llama70']

# Function to fix code using the models
def fix(prompt, model, tokenizer, max_length=200):
    prompt = f"Please provide only the code for the following task, without any comments or explanations.\n\nHere is some code with additional noisy characters inserted:\n\n```{prompt}```\n\nGive me the complete code, without any further explanation:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]
    outputs = model.generate(**inputs, max_new_tokens=max_length)
    generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    return generated_text

# Process each model
for model_name in model_names:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)#.to(device)
    #model = model.bfloat16().cuda()

    fixed_codes = [fix(code, model, tokenizer) for code in tqdm(original_df['Noisy_Code'], desc=f"Fixing code with {model_name}")]
    original_df[f"Fixed Code ({model_name})"] = fixed_codes

    file_name = os.path.join(results_dir, f"result-{model_name.replace('/', '-')}.csv")
    original_df.to_csv(file_name, index=False)

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
            print(f"Warning: Word '{word}' not found in the GloVe model.")
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
                    print(f"Warning: Encountered empty vector at position {i}.")
                    continue
                similarity = cosine_similarity([vec1], [vec2])[0][0]
                sims.append(similarity)
            if sims:
                mean_similarity = np.mean(sims)
                if mean_similarity > max_mean_similarity:
                    max_mean_similarity = mean_similarity
        
        similarities.append(max_mean_similarity)
    
    avg_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)
    return avg_similarity, std_similarity

# Load results
results = [pd.read_csv(os.path.join(results_dir, f"result-{model_name.replace('/', '-')}.csv")) for model_name in model_names]

# Evaluate models
scores, std_devs = [], []
for df, model_name in zip(results, model_names):
    avg_sim, std_sim = calculate_max_cosine_similarity_word_by_word_with_sliding_window(df, 'canonical_solution', f"Fixed Code ({model_name})")
    scores.append(avg_sim)
    std_devs.append(std_sim)

# Evaluate baseline (Noisy_Code)
# Since 'Noisy_Code' is the same across all dataframes, calculate baseline once
df_baseline = results[0]  # Use the first dataframe
avg_sim_base, std_sim_base = calculate_max_cosine_similarity_word_by_word_with_sliding_window(df_baseline, 'canonical_solution', 'Noisy_Code')

# Append the baseline to scores and std_devs
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
ax.set_title('Performance Evaluation after Adding Noise - Large Models with Baseline')
plt.tight_layout()
plt.savefig("plots/performance_evaluation_after_adding_noise_with_baseline-large.png")