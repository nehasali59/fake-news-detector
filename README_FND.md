# 📰 Fake News Detector — Explainable AI System

An AI-powered web application that detects whether a news article or headline is **Real or Fake** and explains the reasoning behind every prediction using **LIME (Explainable AI).**

---

## 🎯 What Makes This Different

Most fake news detectors just say "Fake" or "Real." This project goes further — it shows **exactly which words influenced the prediction** using LIME explainability. This makes the system transparent, trustworthy, and far more useful in real-world scenarios.

---

## 🚀 Features

- Paste any news article or headline and get instant prediction
- Confidence score shown for every prediction
- LIME explanation highlights which words pushed towards Fake or Real
- Clean and modern web interface built with Flask
- Two models built and compared — Logistic Regression and DistilBERT

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| ML Models | Logistic Regression, DistilBERT |
| Explainability | LIME (Local Interpretable Model-Agnostic Explanations) |
| NLP | TF-IDF Vectorizer, HuggingFace Transformers |
| Backend | Flask |
| Frontend | HTML, CSS |
| Dataset | Fake and Real News Dataset (Kaggle) |
| Deep Learning | PyTorch |
| Version Control | GitHub |

---

## 📊 Model Performance

| Model | Accuracy |
|---|---|
| Logistic Regression + TF-IDF | 98.64% |
| DistilBERT (Fine-tuned) | 99.96% |

---

## 📁 Project Structure

```
fake-news-detector/
│
├── dataset/                        ← Dataset folder (not pushed to GitHub)
│   └── cleaned_news.csv
│
├── model/
│   └── saved_model/
│       ├── lr_model.pkl            ← Saved Logistic Regression model
│       ├── tfidf_vectorizer.pkl    ← Saved TF-IDF vectorizer
│       └── distilbert_model/       ← Saved DistilBERT model (not pushed)
│
├── app/
│   ├── app.py                      ← Flask backend
│   ├── templates/
│   │   └── index.html              ← Frontend UI
│   └── static/
│       └── style.css               ← Styling
│
├── explainability/
│   └── lime_explain.py             ← LIME explainability module
│
├── notebooks/
│   ├── EDA.ipynb                   ← Data exploration notebook
│   └── model_training.ipynb        ← Model training notebook
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/nehasali59/fake-news-detector.git
cd fake-news-detector
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the Flask app**
```bash
cd app
python app.py
```

**4. Open in browser**
```
http://127.0.0.1:5000
```

---

## 💡 How It Works

1. User pastes a news article or headline in the web form
2. Flask backend receives the text
3. Text is vectorized using TF-IDF
4. Logistic Regression model predicts Fake or Real
5. Confidence score is calculated
6. LIME generates word-level explanation
7. Result is displayed with prediction, confidence and highlighted words

---

## 🧠 What is LIME?

LIME stands for **Local Interpretable Model-Agnostic Explanations.**

It works by:
- Creating hundreds of modified versions of the input article
- Running each version through the model
- Identifying which words caused the biggest change in prediction
- Highlighting those words as green (Real signals) or red (Fake signals)

This makes the AI system **explainable and transparent** — a critical requirement in real-world AI deployment.

---

## 📚 Dataset

- **Name:** Fake and Real News Dataset
- **Source:** Kaggle
- **Size:** 44,898 articles (23,481 Fake + 21,417 Real)
- **After cleaning:** 39,105 articles
- **Link:** https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

---

## ⚠️ Limitations

- Model trained on 2017-2018 US political news data
- May not perform accurately on sports, entertainment or non-political news
- High accuracy on this dataset due to clean and structured data — real world performance may vary
- Not intended for production use without retraining on diverse, recent data

---

## 🔮 Future Scope

- Retrain on multilingual dataset to support Hindi and other Indian languages
- Integrate DistilBERT model directly into Flask for higher accuracy predictions
- Add URL-based analysis — paste a news link instead of text
- Deploy on cloud platform (AWS / Heroku / Render)
- Add browser extension for real-time news verification

