# 📄 RAGify — Intelligent Document Q&A System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ragify-doc.streamlit.app)

> Powered by Groq + LLaMA 3 + FAISS + LangChain

## 🌐 Live Demo
👉 [ragify-doc.streamlit.app](https://ragify-doc.streamlit.app)

## 📌 About
RAGify is an intelligent document question-answering system built using 
Retrieval-Augmented Generation (RAG). Users can upload any PDF document 
and ask natural language questions to get accurate, source-cited answers.

## ✨ Features
- 📁 **Multi-PDF Support** — Upload multiple PDFs at once
- 🔍 **Hybrid Search** — BM25 + Semantic search for better retrieval
- 🧠 **Conversation Memory** — Remembers last 3 Q&A for context
- 📊 **Confidence Score** — High/Medium/Low answer confidence
- 📋 **Auto Summary** — Document summary on upload
- 💬 **Chat History** — Full conversation history
- 📥 **Export Chat** — Download chat as text file
- 🌐 **Live Deployment** — Accessible from any device

## 🏆 RAGAS Evaluation Results
| Metric | Score |
|--------|-------|
| Faithfulness | 0.90 / 1.00 |
| Answer Relevancy | 0.90 / 1.00 |
| Context Recall | 0.87 / 1.00 |
| **Overall Score** | **0.89 / 1.00** |

## 🛠️ Tech Stack
| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| LLM | LLaMA 3 via Groq API |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB | FAISS |
| Framework | LangChain |
| PDF Parser | PyMuPDF |
| Deployment | Streamlit Cloud |

## 🚀 How to Run Locally
```bash
git clone https://github.com/mamta65-hub/RAGify.git
cd RAGify
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Project Structure
