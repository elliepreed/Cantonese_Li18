# ==========================================
# Install dependencies
# ==========================================
!pip install -q transformers torch huggingface_hub pandas tqdm numpy

# ==========================================
# Imports
# ==========================================
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
import pandas as pd
from tqdm import tqdm

# ==========================================
# Download Cantonese Dataset (CSV) from Hugging Face
# ==========================================
dataset_path = hf_hub_download(
    repo_id="elliepreed/Cantonese_Li18",
    filename="idioms/Cantonese_data.csv",
    repo_type="dataset"
)

df = pd.read_csv(dataset_path, encoding="utf-8")
df.columns = df.columns.str.strip()

required_cols = {"idiom", "correct_translation", "literal_translation"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Missing columns: {required_cols - set(df.columns)}")

print(f"Dataset loaded! Size: {len(df)} rows, Columns: {df.columns.tolist()}")

# ==========================================
# Load CantoneseLLMChat-7B (PyTorch)
# ==========================================
model_id = "hon9kon9ize/CantoneseLLMChat-v1.0-7B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

# ==========================================
# Scoring Function (log-probs)
# ==========================================
def score_answer(model, tokenizer, context, answer):
    device = next(model.parameters()).device
    enc_full = tokenizer(context + answer, return_tensors="pt").to(device)
    enc_context = tokenizer(context, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model(**enc_full)
        logits = output.logits
        answer_ids = enc_full.input_ids[0, len(enc_context.input_ids[0]):]  # slice answer part

        if len(answer_ids) == 0:
            return -100.0  # fallback

        log_probs = torch.log_softmax(logits[0, -len(answer_ids):, :], dim=-1)
        token_logprobs = log_probs.gather(1, answer_ids.unsqueeze(-1)).squeeze(-1)
        return token_logprobs.mean().item()

# ==========================================
# Forced-Choice Evaluation
# ==========================================
results = []
correct_count = 0

for _, row in tqdm(df.iterrows(), total=len(df)):
    idiom = str(row["idiom"]).strip()
    correct_translation = str(row["correct_translation"]).strip()
    literal_translation = str(row["literal_translation"]).strip()

    context = f"Translate the Cantonese idiom into natural English:\n{idiom}\nEnglish: "

    score_correct = score_answer(model, tokenizer, context, correct_translation)
    score_literal = score_answer(model, tokenizer, context, literal_translation)

    margin = score_correct - score_literal
    model_choice = correct_translation if score_correct >= score_literal else literal_translation
    model_was_correct = model_choice == correct_translation

    if model_was_correct:
        correct_count += 1

    # Append to results
    results.append({
        "idiom": idiom,
        "correct_translation": correct_translation,
        "literal_translation": literal_translation,
        "score_correct": score_correct,
        "score_literal": score_literal,
        "margin": margin,
        "model_choice": model_choice,
        "model_was_correct": model_was_correct
    })

    # ===== Print choice for each idiom =====
    print("-----------------------------------")
    print(f"Idiom: {idiom}")
    print(f"Correct Translation: {correct_translation}")
    print(f"Literal Translation: {literal_translation}")
    print(f"Model Choice: {model_choice}")
    print(f"Model Was Correct: {model_was_correct}")
    print(f"Score (Correct): {score_correct:.4f}")
    print(f"Score (Literal): {score_literal:.4f}")
    print(f"Margin: {margin:.4f}")
    print("-----------------------------------")

# ==========================================
# Final Accuracy
# ==========================================
accuracy = correct_count / len(df)
print("\n===================================")
print("FINAL FORCED-CHOICE ACCURACY:", round(accuracy, 4))
print("===================================\n")

# ==========================================
# Save Results
# ==========================================
results_df = pd.DataFrame(results)
results_df.to_csv("cantonese_forced_choice_results.csv", index=False)
print("Results saved to cantonese_forced_choice_results.csv")
