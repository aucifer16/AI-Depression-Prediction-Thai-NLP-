# -------------------------------------------------------------------
# 🔒 Safe Naive Bayes Text Classification (No pickle)
# This script loads a Naive Bayes model and vocabulary from JSON files,
# predicts labels for text in a CSV file, and saves the results.
# -------------------------------------------------------------------

import json, math
import pandas as pd
from pythainlp.tokenize import word_tokenize  # Thai word tokenizer

# ---------- ⚙️ File paths ----------
MODEL_JSON = "finalized_model_nltk_nb.json"   # JSON model (converted from pickle)
VOCAB_JSON = "vocabulary_words.json"          # Vocabulary file
INPUT_CSV  = "28_08_2023 17_35_12.csv"        # Input text CSV
OUTPUT_CSV = "28_08_2023 17_35_12P.csv"       # Output CSV with predictions

# ---------- 📦 Load the Naive Bayes model safely from JSON ----------
def load_safe_nb(json_path: str) -> dict:
    """Load Naive Bayes model parameters from a safe JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        M = json.load(f)
    # Check the model type
    assert M.get("model_type") == "nltk_naive_bayes", "Invalid model format"
    # Convert all numeric values to float for consistency
    M["label_priors"] = {k: float(v) for k, v in M["label_priors"].items()}
    for lbl, d in M["feature_probs"].items():
        for fname, bucket in d.items():
            for val, p in list(bucket.items()):
                bucket[val] = float(p)
    M["smoothing_epsilon"] = float(M.get("smoothing_epsilon", 1e-12))
    return M

# ---------- 🤖 Compute class probabilities ----------
def predict_proba_safe_nb(M: dict, features: dict) -> dict:
    """Return class probabilities for a feature dictionary."""
    labels = M["labels"]                # All class labels
    priors = M["label_priors"]          # Prior probabilities
    fprobs = M["feature_probs"]         # Feature likelihoods
    eps = float(M["smoothing_epsilon"]) # Smoothing factor to prevent zero

    logps = []
    for lbl in labels:
        lp = math.log(max(priors.get(lbl, eps), eps))
        # Iterate through all features (True/False for each word)
        for fname, fval in features.items():
            bucket = fprobs.get(lbl, {}).get(fname, {})
            # Get P(feature=value | label), or use epsilon if missing
            p = float(bucket.get(str(fval), eps))
            lp += math.log(max(p, eps))
        logps.append(lp)

    # Convert log-probabilities to normalized probabilities (softmax)
    m = max(logps)
    exps = [math.exp(lp - m) for lp in logps]
    Z = sum(exps) or 1.0
    probs = [e / Z for e in exps]
    return dict(zip(labels, probs))

# ---------- 🏷️ Predict the most likely class ----------
def predict_safe_nb(M: dict, features: dict):
    """Return the predicted label and probability distribution."""
    probs = predict_proba_safe_nb(M, features)
    return max(probs.items(), key=lambda kv: kv[1])[0], probs

# ---------- 📚 Load the vocabulary ----------
def load_vocabulary(path: str):
    """Load vocabulary list from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    # Ensure it's a list (in case it was saved as a set)
    return list(vocab)

# ---------- ✍️ Convert text → features ----------
def featurize(text: str, vocabulary) -> dict:
    """
    Convert a text string into a Boolean feature dictionary:
    Each vocabulary word becomes a key, True if present in text.
    """
    tokens = set(word_tokenize(str(text).lower()))
    return {w: (w in tokens) for w in vocabulary}

# ---------- 🚀 Main process ----------
def main():
    print("Start Program (safe JSON mode)")
    M = load_safe_nb(MODEL_JSON)           # Load model from JSON
    vocabulary = load_vocabulary(VOCAB_JSON)  # Load vocabulary
    print("Run Predict")

    df = pd.read_csv(INPUT_CSV)            # Read input CSV
    cnt = 0

    # Function to predict for each row of text
    def _predict_one(txt):
        nonlocal cnt
        cnt += 1
        if cnt % 100 == 0:
            print(f"Processed {cnt} rows...")
        feats = featurize(txt, vocabulary)     # Convert text to features
        yhat, _ = predict_safe_nb(M, feats)    # Predict label
        return yhat

    # Apply prediction to each text row
    df["Result"] = df["text"].apply(_predict_one)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print("✅ Done! Results saved to:", OUTPUT_CSV)

# ---------- 🧭 Entry point ----------
if __name__ == "__main__":
    main()