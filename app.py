import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
import tempfile
import os

st.set_page_config(page_title="RAGify", page_icon="📄", layout="wide")

st.markdown('<style>.main{background:#f8f9fa}</style>', unsafe_allow_html=True)

st.markdown('<div style="background:linear-gradient(135deg,#667eea,#764ba2);padding:2rem;border-radius:15px;text-align:center;color:white;margin-bottom:2rem"><h1>📄 RAGify</h1><p>Intelligent Document Q&A — Powered by Groq + LLaMA 3</p></div>', unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0
if "client" not in st.session_state:
    st.session_state.client = None

with st.sidebar:
    st.markdown("## Settings")
    groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    if groq_key:
        st.session_state.client = Groq(api_key=groq_key)
        st.success("Connected!")
    st.markdown("---")
    st.markdown("## Upload Documents")
    uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
    if uploaded_files and groq_key:
        if st.button("Process Documents", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
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
                st.session_state.vectorstore = FAISS.from_documents(all_chunks, embeddings)
                st.session_state.total_chunks = len(all_chunks)
                sample = " ".join([c.page_content for c in all_chunks[:5]])
                resp = st.session_state.client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": "Summarize in 3 sentences:\n" + sample}],
                    max_tokens=150
                )
                st.session_state.summary = resp.choices[0].message.content
                st.session_state.chat_history = []
            st.success("Done! " + str(len(all_chunks)) + " chunks ready!")
            st.rerun()
    if st.session_state.vectorstore:
        st.markdown("---")
        st.markdown("## Stats")
        st.metric("Chunks Indexed", st.session_state.total_chunks)
        st.metric("Questions Asked", len(st.session_state.chat_history))
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

if not st.session_state.vectorstore:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("Step 1 - Enter Groq API Key")
    with c2:
        st.info("Step 2 - Upload PDF files")
    with c3:
        st.info("Step 3 - Ask questions!")
    st.markdown("---")
    st.markdown("### Features")
    f1, f2, f3, f4, f5 = st.columns(5)
    with f1: st.success("Multi-PDF")
    with f2: st.success("LLaMA 3")
    with f3: st.success("Confidence")
    with f4: st.success("Chat History")
    with f5: st.success("Auto Summary")
else:
    if st.session_state.summary:
        with st.expander("Document Summary", expanded=False):
            st.write(st.session_state.summary)
    st.markdown("### Conversation")
    for item in st.session_state.chat_history:
        st.markdown('<div style="background:#667eea;color:white;padding:0.8rem;border-radius:18px 18px 4px 18px;margin:0.5rem 0;max-width:80%;margin-left:auto">You: ' + item["question"] + '</div>', unsafe_allow_html=True)
        st.markdown('<div style="background:white;padding:0.8rem;border-radius:18px 18px 18px 4px;margin:0.5rem 0;max-width:80%;border:1px solid #e0e0e0">RAGify: ' + item["answer"] + '<br><small>Confidence: ' + item["confidence"] + '</small></div>', unsafe_allow_html=True)
        st.markdown('<div style="background:#f8f9fa;padding:0.5rem;border-radius:8px;border-left:4px solid #667eea;font-size:0.8rem;color:#666">Source: ' + item["source"] + '...</div>', unsafe_allow_html=True)
    st.markdown("---")
    col1, col2 = st.columns([5, 1])
    with col1:
        question = st.text_input("", placeholder="Ask anything about your document...", label_visibility="collapsed")
    with col2:
        ask = st.button("Ask", type="primary", use_container_width=True)
    if ask and question and st.session_state.client:
        with st.spinner("Thinking..."):
            results = st.session_state.vectorstore.similarity_search_with_score(question, k=3)
            scores = [s for _, s in results]
            avg = sum(scores) / len(scores)
            confidence = "High" if avg < 0.3 else "Medium" if avg < 0.6 else "Low"
            context = "\n\n".join([d.page_content for d, _ in results])
            response = st.session_state.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": "Answer using only this context:\n" + context + "\n\nQuestion: " + question + "\nAnswer:"}],
                temperature=0.3,
                max_tokens=300
            )
            answer = response.choices[0].message.content
            st.session_state.chat_history.append({
                "question": question,
                "answer": answer,
                "confidence": confidence,
                "source": results[0][0].page_content[:250]
            })
            st.rerun()
