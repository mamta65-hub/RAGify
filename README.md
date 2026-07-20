# 📄 RAGify — Intelligent Document Q&A System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ragify-doc.streamlit.app)

> Powered by Groq + LLaMA 3 + FAISS + Pinecone + LangChain

## 🌐 Live Demo
👉 [ragify-doc.streamlit.app](https://ragify-doc.streamlit.app)

## 📌 About
RAGify is an intelligent document question-answering system built using Retrieval-Augmented Generation (RAG). Users can upload PDF documents or provide URLs and ask natural language questions to get accurate, source-cited answers powered by LLaMA 3.

## ✨ Features

### 🔥 Level 1 — Core Features
- 🌙 **Dark Mode** — Beautiful dark/light theme toggle
- 🌐 **Multi-Language** — Answers in English, Hindi, or Hinglish
- 🎤 **Voice Input** — Ask questions by speaking
- 📋 **Copy Answer** — One-click answer copying

### 🔥 Level 2 — Advanced Features
- 🔍 **OCR Support** — Scanned PDF processing
- 📊 **Table Extraction** — Extract and query PDF tables
- 🌐 **URL Support** — Query any website content
- 📄 **Multi-Language PDF** — Hindi & English PDFs

### 🔥 Level 3 — Pro Features
- 🌐 **Persistent Storage** — Pinecone vector database
- 🔐 **User Authentication** — Secure login system
- 📝 **Quiz Generator** — Auto MCQ generation from PDFs
- 🃏 **Flashcard Generator** — Study flashcards

### 🔥 Level 4 — Research Grade
- 🔬 **Chunking Strategy Comparison** — Compare retrieval strategies
- 🤖 **Model Comparison** — LLaMA vs Gemma vs Mixtral
- 🎯 **Domain-Specific Prompts** — Medical, Legal, CS, Finance, Research

## 🏆 RAGAS Evaluation Results

| Metric | Score |
|--------|-------|
| Faithfulness | 0.90 / 1.00 |
| Answer Relevancy | 0.90 / 1.00 |
| Context Recall | 0.84 / 1.00 |
| **Overall Score** | **0.88 / 1.00** |

> 🌟 **EXCELLENT — Publication Ready!**

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| LLM | LLaMA 3 / Gemma / Mixtral via Groq API |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB | FAISS (local) + Pinecone (persistent) |
| Framework | LangChain |
| PDF Parser | PyMuPDF + OCR (Tesseract) |
| Authentication | Custom Login System |
| Deployment | Streamlit Cloud |

## 🚀 How to Run Locally

```bash
git clone https://github.com/mamta65-hub/RAGify.git
cd RAGify
pip install -r requirements.txt
streamlit run app.py
```

## 🔑 Setup
1. Get free Groq API key from [console.groq.com](https://console.groq.com)
2. Login with demo credentials
3. Enter Groq API key in sidebar
4. Upload PDF or enter URL
5. Start asking questions!

## 📁 Project Structure
