# ==========================================
# Install dependencies
# ==========================================
!pip install -q llama-cpp-python huggingface_hub pandas tqdm numpy

# ==========================================
# Imports
# ==========================================
from huggingface_hub import hf_hub_download
import pandas as pd
import numpy as np
from tqdm import tqdm
from llama_cpp import Llama

# ==========================================
# Download Dataset
# ==========================================
dataset_path = hf_hub_download(
    repo_id="elliepreed/Malay_Li18",
    filename="idioms/Malay_data.tsv",
    repo_type="dataset"
)

df = pd.read_csv(dataset_path)
df.columns = df.columns.str.strip()   # remove leading/trailing spaces

print("Dataset size:", len(df))
print("Columns:", df.columns)
# ==========================================
# Download GGUF Model
# (Adjust filename if needed)
# ==========================================
model_path = hf_hub_download(
    repo_id="Chemin-AI/malaysian-Llama-3.2-3B-Instruct-gguf",
    filename="malaysian-Llama-3.2-3B-Instruct.q4_k_m.gguf"
)

print("Model path:", model_path)

# ==========================================
# Load Model
# ==========================================
llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    logits_all=True,
    verbose=False
)

# ==========================================
# Scoring Function
# Computes log P(answer | context)
# Length-normalized
# ==========================================
def score_answer(llm, context, answer):

    full_prompt = context + answer

    # Get full prompt logprobs
    response = llm.create_completion(
        prompt=full_prompt,
        max_tokens=0,
        echo=True,
        logprobs=1
    )

    tokens = response["choices"][0]["logprobs"]["tokens"]
    token_logprobs = response["choices"][0]["logprobs"]["token_logprobs"]

    # Get context-only logprobs to find answer start
    context_response = llm.create_completion(
        prompt=context,
        max_tokens=0,
        echo=True,
        logprobs=1
    )

    context_length = len(
        context_response["choices"][0]["logprobs"]["tokens"]
    )

    # Only score answer tokens
    answer_logprobs = token_logprobs[context_length:]
    answer_logprobs = [lp for lp in answer_logprobs if lp is not None]

    if len(answer_logprobs) == 0:
        # Safety: if no tokens were scored, return -inf
        print("⚠️ Warning: no logprobs for answer:", answer)
        return -float("inf")

    # Return length-normalized log probability
    return sum(answer_logprobs) / len(answer_logprobs)
# ==========================================
# Forced-Choice Evaluation
# ==========================================
results = []
correct_count = 0

for _, row in tqdm(df.iterrows(), total=len(df)):

    idiom = row["idiom"]
    correct_translation = row["correct_translation"]
    literal_translation = row["literal_translation"]

    context = f"Translate the Cantonese idiom into natural English:\n{idiom}\nEnglish: "

    score_correct = score_answer(llm, context, correct_translation)
    score_literal = score_answer(llm, context, literal_translation)

    model_choice = (
        correct_translation
        if score_correct > score_literal
        else literal_translation
    )

    is_correct = score_correct > score_literal

    if is_correct:
        correct_count += 1

    results.append({
        "idiom": idiom,
        "correct_translation": correct_translation,
        "literal_translation": literal_translation,
        "score_correct": score_correct,
        "score_literal": score_literal,
        "model_choice": model_choice,
        "model_was_correct": is_correct
    })

# ==========================================
# Final Accuracy
# ==========================================
accuracy = correct_count / len(df)

print("\n===================================")
print("FINAL FORCED-CHOICE ACCURACY:", accuracy)
print("===================================\n")

# ==========================================
# Print Results Cleanly Per Idiom
# ==========================================
for r in results:
    print("Idiom:", r["idiom"])
    print("Correct Translation:", r["correct_translation"])
    print("Literal Translation:", r["literal_translation"])
    print("Model Chose:", r["model_choice"])
    print("Model Was Correct:", r["model_was_correct"])
    print("Score (Correct):", round(r["score_correct"], 4))
    print("Score (Literal):", round(r["score_literal"], 4))
    print("-----------------------------------")

# Optional: convert to dataframe
results_df = pd.DataFrame(results)
