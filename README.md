# 🧠 AI Depression Prediction (Thai NLP) — Safe Naive Bayes (JSON Version)

This repository contains the improved and secure version of the **AI-based depression risk prediction** system for **Thai** social media texts.  
It uses **Natural Language Processing (NLP)** and the **Naive Bayes Classifier**, with all model data stored in **JSON format** for safety (no `pickle` execution).

> **Main script:** `safe_run_nb.py`  
> Loads a pre-trained JSON model (`finalized_model_nltk_nb.json`) and `vocabulary_words.json`, reads an input CSV with a `text` column, and outputs predictions to a new CSV with a `Result` column.

---

## ✨ Features
- **Thai text tokenization** using [PyThaiNLP](https://github.com/PyThaiNLP/pythainlp)
- **Bag-of-Words** feature extraction (`{word: True/False}`)
- **Naive Bayes classification** for depression-risk detection
- **CSV input/output pipeline** — ready for batch text prediction
- **Safe & secure**: all model data are stored as JSON, not pickle

---

## 📂 Project Structure
```
.
├── safe_run_nb.py                        # Main prediction script (safe version)
├── finalized_model_nltk_nb.json          # Pre-trained model (JSON)
├── vocabulary_words.json                 # Vocabulary list (JSON)
├── 28_08_2023 17_35_12.csv               # Input CSV (must include 'text' column)
├── 28_08_2023 17_35_12P.csv              # Output CSV with predictions
└── README.md
```

> Input/output filenames are defined inside `safe_run_nb.py`.  
> You can edit them to match your own dataset.

---

## ⚙️ Installation
Requires **Python 3.9+**

```bash
# 1) Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Install dependencies
pip install nltk pythainlp pandas

# 3) Download tokenizer data for NLTK (if not already installed)
python -m nltk.downloader punkt
```

---

## 🚀 Usage

Make sure the following files exist in the same directory:
- `finalized_model_nltk_nb.json` — pre-trained Naive Bayes model  
- `vocabulary_words.json` — list of vocabulary used for feature extraction  
- `28_08_2023 17_35_12.csv` — input dataset (must include column `text`)

Then run:
```bash
python safe_run_nb.py
```

### The script will:
1. Load the model and vocabulary from JSON  
2. Tokenize each sentence with PyThaiNLP  
3. Convert text into bag-of-words features  
4. Predict the depression risk class  
5. Save the results to a new CSV file  

#### Input example
```csv
text
"รู้สึกเหนื่อยมากเลยวันนี้"
"วันนี้อากาศดีจัง"
```

#### Output example
```csv
text,Result
"รู้สึกเหนื่อยมากเลยวันนี้",negative
"วันนี้อากาศดีจัง",positive
```

---

## 🧠 Model Details
- Based on **NLTK NaiveBayesClassifier**
- Features: `{word: True/False}` per tokenized word
- Tokenizer: `pythainlp.tokenize.word_tokenize`
- Stored as **JSON** instead of pickle for safer deployment

### Original Model Performance
(from the published research paper)
| Metric | Value |
|:--|:--|
| Training Accuracy | 88.17% |
| Testing Accuracy | 75.00% |
| F1 Score (Negative class) | 76.70% |

> The model performs best for **negative (depressive)** messages  
> and has moderate accuracy for **neutral or sarcastic** content.

---

## 📊 Dataset Summary
- Collected from **Twitter (X)** hashtags:  
  `#โรคซึมเศร้า`, `#ซึมเศร้า`, `#คนเก่ง`, `#สู้ต่อไป`
- Manually labeled by **three psychiatrists**
- Split: **70% training**, **30% testing**
- Total size: ~15,000 posts

---

## 🔐 Ethics and Privacy
- All data were **anonymized** under Thailand’s **PDPA**  
- This AI model is a **screening tool**, not a clinical diagnostic  
- Should not be used for medical decision-making without psychiatric evaluation

---

## 👩‍💻 Authors
- **Patcharin Boonsomthop** — National Institute of Development Administration (NIDA)  
  ✉️ [patcharin.b@kru.ac.th](mailto:patcharin.b@kru.ac.th)

- **Chutisant Kerdvibulvech** — National Institute of Development Administration (NIDA)  
  ✉️ [chutisant.k@nida.ac.th](mailto:chutisant.k@nida.ac.th)

---

## 📚 Citation
If you use this repository in your research, please cite:

```
Boonsomthop, P., & Kerdvibulvech, C. (2025).
Development of an Innovative Media Model Using Artificial Intelligence for Predicting Depression.
Journal of Advanced Computational Intelligence and Intelligent Informatics (JACIII), 29(5), 1126–1131.
```

---

## 🧾 License
This project is released for **academic and educational purposes only**.  
Commercial use, redistribution, or identification of social media users is **strictly prohibited**.

---

> ⚠️ **Disclaimer:**  
> This AI model serves as an early screening tool for detecting potential depression indicators in social media posts.  
> It should **not** be used as a substitute for professional diagnosis or therapy.
