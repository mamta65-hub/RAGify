import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
from datetime import datetime
import tempfile
import os

st.set_page_config(page_title="RAGify", page_icon="📄", layout="wide")

st.markdown('<style>.main{background:#f8f9fa}.stButton button{border-radius:20px}</style>', unsafe_allow_html=True)
st.markdown('<div style="background:linear-gradient(135deg,#667eea,#764ba2);padding:2rem;border-radius:15px;text-align:center;color:white;margin-bottom:2rem"><h1>📄 RAGify</h1><p>Intelligent Document Q&A — Powered by Groq + LLaMA 3</p></div>', unsafe_allow_html=True)

for key in ["chat_history","vectorstore","summary","total_chunks","client","all_chunks"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key in ["chat_history","all_chunks"] else None
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    groq_key = st.text_input("🔑 Groq API Key", type="password", placeholder="gsk_...")
    if groq_key:
        st.session_state.client = Groq(api_key=groq_key)
        st.success("✅ Connected!")
    else:
        st.warning("⚠️ Enter Groq API Key")
    st.markdown("---")
    st.markdown("## 📁 Upload Documents")
    uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
    if uploaded_files and groq_key:
        if st.button("🚀 Process Documents", type="primary", use_container_width=True):
            with st.spinner("⏳ Processing..."):
                all_chunks = []
                for f in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(f.read())
                        tmp_path = tmp.name
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                    chunks = splitter.split_documents(docs)
                    all_chunks.extend(chunks)
                    os.unlink(tmp_path)
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vectorstore = FAISS.from_documents(all_chunks, embeddings)
                st.session_state.vectorstore = vectorstore
                st.session_state.all_chunks = all_chunks
                st.session_state.total_chunks = len(all_chunks)
                sample = " ".join([c.page_content for c in all_chunks[:5]])
                resp = st.session_state.client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": "Summarize in 3 sentences:\n" + sample}],
                    max_tokens=150
                )
                st.session_state.summary = resp.choices[0].message.content
                st.session_state.chat_history = []
            st.success("✅ " + str(len(all_chunks)) + " chunks ready!")
            st.rerun()
    if st.session_state.vectorstore:
        st.markdown("---")
        st.markdown("## 📊 Stats")
        st.metric("Chunks Indexed",st.session_state.total_chunks
        st.metric("Questions Asked",len(st.session_state.chat_history))
