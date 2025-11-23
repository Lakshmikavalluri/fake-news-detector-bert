# 📰 Fake News Detector — BERT Powered (Streamlit + Transformers)

A powerful NLP-based Fake News Detection system using **BERT (Transformer model)**, fine-tuned on Real + Fake news datasets including additional **Indian news samples** for better real-world accuracy.  
The project includes training, prediction, and a beautiful **Streamlit web UI** for interacting with the model.

---

## 🚀 Features

- 🔥 **BERT (bert-base-uncased)** for high-accuracy text classification  
- 🧠 Detects **Real vs Fake** news with confidence score  
- 🇮🇳 Optimized for **Indian news** (ISRO, RBI, Govt announcements)  
- 🧹 Clean data processing, tokenization & transformer pipeline  
- 🌐 Modern **Streamlit UI** with card layout and confidence bar  
- ☁️ Deployable on **Streamlit Cloud**  
- 📁 Full training + inference scripts included  

---

## 📦 Tech Stack

- **Python 3.10+**
- **PyTorch**
- **HuggingFace Transformers**
- **Streamlit**
- **Scikit-Learn**
- **Pandas / NumPy**

---

## 📂 Project Structure
fake-news-detector-bert/
│
├── data/
│ └── news.csv
│ └── news_dataset.csv
│ └── indian_fake_news_dataset.csv # optional (extra)
│
├── model/
│ └── bert/
│ ├── config.json
│ ├── pytorch_model.bin # BERT weights (not uploaded to GitHub)
│ ├── tokenizer.json
│
├── src/
│ ├── train_bert.py # BERT training script
│ ├── predict_bert.py # Prediction script
│ ├── streamlit_app.py # Streamlit UI
│
├── .gitignore # Prevents large model files from being tracked
├── requirements.txt
└── README.md
