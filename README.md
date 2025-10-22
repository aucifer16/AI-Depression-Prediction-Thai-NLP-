# 🧠 AI Depression Prediction (Thai NLP) — Naive Bayes

This repository contains the code and research work for **AI-based depression risk prediction** in **Thai** social media texts using **Natural Language Processing (NLP)** and the **Naive Bayes Classifier**.

> **Main script:** `runPredictAll.py`  
> Loads a pre-trained model (`finalized_model.pickle`) and `vocabulary_model.pickle`, reads an input CSV with a `text` column, and outputs predictions to a new CSV with a `Result` column.

---

## ✨ Features
- Thai text tokenization via **PyThaiNLP**
- Feature extraction using a **Bag-of-Words** vocabulary
- Sentiment / depression-risk classification (positive / negative / neutral)
- Easy CSV input and output workflow

---

## 📂 Project Structure
```
.
├── runPredictAll.py                      # Main prediction script
├── finalized_model.pickle                # Pre-trained Naive Bayes model
├── vocabulary_model.pickle               # Vocabulary/features for the model
├── usertwitter_iamsobad15feed.csv        # Input CSV (must include 'text' column)
├── usertwitter_iamsobad15feed_Predit.csv # Output CSV with predictions (auto-generated)
└── README.md
```

> Input/output file names are defined inside `runPredictAll.py`.  
> Edit the script if you want to use custom filenames.

---

## ⚙️ Installation
Requires **Python 3.9+**

```bash
# 1) Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

# 2) Install dependencies
pip install nltk pythainlp pandas

# 3) Download tokenizer data for NLTK
python -m nltk.downloader punkt
```

---

## 🚀 Usage

Make sure the following files exist in the same folder:
- `finalized_model.pickle` — trained classifier
- `vocabulary_model.pickle` — word features
- `usertwitter_iamsobad15feed.csv` — input dataset (must include column `text`)

Then run:
```bash
python runPredictAll.py
```

### The script will:
1. Load the pre-trained model and vocabulary  
2. Tokenize and featurize each text line  
3. Predict the depression risk class  
4. Write results to `usertwitter_iamsobad15feed_Predit.csv`

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
- Features: `{word: True/False}` presence derived from tokenized text
- Tokenizer: `pythainlp.tokenize.word_tokenize`
- Performance (from the paper):
  - **Training accuracy:** 88.17%
  - **Testing accuracy:** 75.00%
  - **F1 (Negative class):** 76.70%

> The model performs best on **negative (depressive)** messages and has limited accuracy for **neutral or sarcastic** text.

---

## 📊 Dataset Summary
- Data collected from **Twitter (X)** hashtags such as  
  `#โรคซึมเศร้า`, `#ซึมเศร้า`, `#คนเก่ง`, `#สู้ต่อไป`
- Reviewed and labeled by **three psychiatrists**
- Split: **70% training**, **30% testing**
- ~15,000 total social media posts

---

## 🔐 Ethics and Privacy
- All data were anonymized in compliance with Thailand’s **PDPA (Personal Data Protection Act)**  
- The AI model is a **screening tool only** and not a diagnostic system  
- Should not be used for clinical decisions without psychiatric review

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
> This model serves as an early warning tool for detecting possible depression indicators in social media posts.  
> It should **not** be used to replace professional medical diagnosis or treatment.
