import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
from datetime import datetime
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
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []

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
            with st.spinner("⏳ Processing... Please wait!"):
                all_chunks = []
                for f in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(f.read())
                        tmp_path = tmp.name
                    try:
                        loader = PyMuPDFLoader(tmp_path)
                        docs = loader.load()
                        st.write("Pages loaded: " + str(len(docs)))
                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=500,
                            chunk_overlap=50
                        )
                        chunks = splitter.split_documents(docs)
                        all_chunks.extend(chunks)
                        st.write("Chunks created: " + str(len(chunks)))
                    except Exception as e:
                        st.error("Error: " + str(e))
                    finally:
                        os.unlink(tmp_path)

                if all_chunks:
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
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
                else:
                    st.error("No chunks created! PDF might be scanned/image-based.")
            st.rerun()

    if st.session_state.vectorstore:
        st.markdown("---")
        st.markdown("## 📊 Stats")
        st.metric("Chunks Indexed", st.session_state.total_chunks)
        st.metric("Questions Asked", len(st.session_state.chat_history))
        st.markdown("---")
        if st.button("📥 Export Chat", use_container_width=True):
            if st.session_state.chat_history:
                export_text = "RAGify Chat Export\n" + "="*50 + "\n\n"
                for i, item in enumerate(st.session_state.chat_history, 1):
                    export_text += "Q" + str(i) + ": " + item["question"] + "\n"
                    export_text += "A: " + item["answer"] + "\n"
                    export_text += "Confidence: " + item["confidence"] + "\n"
                    export_text += "-"*30 + "\n\n"
                st.download_button("⬇️ Download", export_text, "ragify_chat.txt", use_container_width=True)
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

if not st.session_state.vectorstore:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**Step 1**\n\n🔑 Enter Groq API Key")
    with c2:
        st.info("**Step 2**\n\n📁 Upload PDF files")
    with c3:
        st.info("**Step 3**\n\n💬 Ask questions!")
    st.markdown("---")
    st.markdown("### ✨ Pro Features")
    f1, f2, f3, f4, f5 = st.columns(5)
    with f1: st.success("🔍 Hybrid Search")
    with f2: st.success("🧠 Memory")
    with f3: st.success("📊 Confidence")
    with f4: st.success("💬 Chat History")
    with f5: st.success("📥 Export Chat")
else:
    if st.session_state.summary:
        with st.expander("📋 Document Summary", expanded=False):
            st.write(st.session_state.summary)
    st.markdown("### 💬 Conversation")
    for item in st.session_state.chat_history:
        st.markdown('<div style="background:#667eea;color:white;padding:0.8rem 1.2rem;border-radius:18px 18px 4px 18px;margin:0.5rem 0;max-width:80%;margin-left:auto;text-align:right">❓ ' + item["question"] + '</div>', unsafe_allow_html=True)
        conf_color = "green" if item["confidence"] == "High" else "orange" if item["confidence"] == "Medium" else "red"
        st.markdown('<div style="background:white;padding:0.8rem 1.2rem;border-radius:18px 18px 18px 4px;margin:0.5rem 0;max-width:80%;border:1px solid #e0e0e0;box-shadow:0 2px 5px rgba(0,0,0,0.1)">🤖 ' + item["answer"] + '<br><small style="color:' + conf_color + '">Confidence: ' + item["confidence"] + ' | ' + item["time"] + '</small></div>', unsafe_allow_html=True)
        st.markdown('<div style="background:#f8f9fa;padding:0.5rem 0.8rem;border-radius:8px;border-left:4px solid #667eea;font-size:0.8rem;color:#666;margin-bottom:1rem">📄 ' + item["source"] + '...</div>', unsafe_allow_html=True)
    st.markdown("---")
    col1, col2 = st.columns([5, 1])
    with col1:
        question = st.text_input("", placeholder="Ask anything about your document...", label_visibility="collapsed")
    with col2:
        ask = st.button("Ask 🔍", type="primary", use_container_width=True)
    if ask and question and st.session_state.client:
        with st.spinner("🤖 Thinking..."):
            semantic_docs = st.session_state.vectorstore.similarity_search(question, k=3)
            question_words = set(question.lower().split())
            bm25_scores = []
            for chunk in st.session_state.all_chunks:
                words = set(chunk.page_content.lower().split())
                score = len(question_words & words)
                bm25_scores.append((score, chunk))
            bm25_scores.sort(key=lambda x: x[0], reverse=True)
            bm25_docs = [doc for _, doc in bm25_scores[:3]]
            seen = set()
            docs = []
            for doc in semantic_docs + bm25_docs:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    docs.append(doc)
            docs = docs[:4]
            score_docs = st.session_state.vectorstore.similarity_search_with_score(question, k=3)
            scores = [s for _, s in score_docs]
            avg = sum(scores) / len(scores)
            confidence = "High" if avg < 0.3 else "Medium" if avg < 0.6 else "Low"
            memory_context = ""
            if len(st.session_state.chat_history) > 0:
                last_3 = st.session_state.chat_history[-3:]
                for item in last_3:
                    memory_context += "Q: " + item["question"] + "\nA: " + item["answer"] + "\n\n"
            context = "\n\n".join([d.page_content for d in docs])
            prompt = "Previous conversation:\n" + memory_context + "\nDocument context:\n" + context + "\n\nCurrent question: " + question + "\n\nAnswer:"
            response = st.session_state.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Answer based on document context. Remember previous conversation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=400
            )
            answer = response.choices[0].message.content
            st.session_state.chat_history.append({
                "question": question,
                "answer": answer,
                "confidence": confidence,
                "source": docs[0].page_content[:250],
                "time": datetime.now().strftime("%H:%M")
            })
            st.rerun()
