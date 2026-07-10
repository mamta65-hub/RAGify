import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
import tempfile
import os

st.set_page_config(page_title="RAGify", page_icon="📄", layout="wide")

st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem; border-radius: 15px;
    text-align: center; color: white; margin-bottom: 2rem;
}
.chat-user {
    background: #667eea; color: white;
    padding: 0.8rem 1.2rem; border-radius: 18px 18px 4px 18px;
    margin: 0.5rem 0; max-width: 80%; margin-left: auto; text-align: right;
}
.chat-bot {
    background: white; color: #333;
    padding: 0.8rem 1.2rem; border-radius: 18px 18px 18px 4px;
    margin: 0.5rem 0; max-width: 80%;
    border: 1px solid #e0e0e0; box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}
.source-box {
    background: #f8f9fa; padding: 0.8rem;
    border-radius: 8px; border-left: 4px solid #667eea;
    font-size: 0.8rem; color: #666; margin-top: 0.3rem;
}
.metric-card {
    background: white; padding: 1rem;
    border-radius: 10px; text-align: center;
    box-shadow: 0
