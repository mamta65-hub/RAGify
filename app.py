import streamlit as st

# ── SIMPLE CUSTOM AUTH ──
USERS = {
    "mamta": {"password": "mamta123", "name": "Mamta"},
    "admin": {"password": "admin123", "name": "Admin"},
    "guest": {"password": "guest123", "name": "Guest"}
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "current_user" not in st.session_state:
    st.session_state.current_user = ""

if not st.session_state.logged_in:
    st.markdown('<div style="max-width:400px;margin:100px auto;padding:2rem;border-radius:15px;box-shadow:0 4px 20px rgba(0,0,0,0.1);background:white">', unsafe_allow_html=True)
    st.markdown("## 🔐 RAGify Login")
    username = st.text_input("👤 Username")
    password = st.text_input("🔑 Password", type="password")
    if st.button("Login →", type="primary", use_container_width=True):
        if username in USERS and USERS[username]["password"] == password:
            st.session_state.logged_in = True
            st.session_state.current_user = USERS[username]["name"]
            st.rerun()
        else:
            st.error("❌ Wrong username or password")
    st.stop()
    st.sidebar.success("✅ Welcome, " + st.session_state.current_user + "!")
    if st.sidebar.button("🚪 Logout"):
       st.session_state.logged_in = False
       st.session_state.current_user = ""
       st.rerun()
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from groq import Groq
from datetime import datetime
from streamlit_mic_recorder import mic_recorder
from langchain_core.documents import Document
import tempfile
import os
import requests
from bs4 import BeautifulSoup
from PIL import Image
import pytesseract
import fitz

st.set_page_config(page_title="RAGify", page_icon="📄", layout="wide")

# Secrets
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")
PINECONE_INDEX = "ragify-index"

def get_css(dark_mode):
    if dark_mode:
        return """<style>
.stApp {background:#1a1a2e;color:#e0e0e0}
.stSidebar {background:#16213e}
.main-header {background:linear-gradient(135deg,#667eea,#764ba2);padding:2rem;border-radius:15px;text-align:center;color:white;margin-bottom:2rem}
.chat-user {background:#667eea;color:white;padding:0.8rem 1.2rem;border-radius:18px 18px 4px 18px;margin:0.5rem 0;max-width:80%;margin-left:auto;text-align:right}
.chat-bot {background:#2d2d44;color:#e0e0e0;padding:0.8rem 1.2rem;border-radius:18px 18px 18px 4px;margin:0.5rem 0;max-width:80%;border:1px solid #444}
.source-box {background:#252540;padding:0.5rem 0.8rem;border-radius:8px;border-left:4px solid #667eea;font-size:0.8rem;color:#aaa;margin-bottom:1rem}
</style>"""
    else:
        return """<style>
.stApp {background:#f8f9fa}
.main-header {background:linear-gradient(135deg,#667eea,#764ba2);padding:2rem;border-radius:15px;text-align:center;color:white;margin-bottom:2rem}
.chat-user {background:#667eea;color:white;padding:0.8rem 1.2rem;border-radius:18px 18px 4px 18px;margin:0.5rem 0;max-width:80%;margin-left:auto;text-align:right}
.chat-bot {background:white;color:#333;padding:0.8rem 1.2rem;border-radius:18px 18px 18px 4px;margin:0.5rem 0;max-width:80%;border:1px solid #e0e0e0;box-shadow:0 2px 5px rgba(0,0,0,0.1)}
.source-box {background:#f8f9fa;padding:0.5rem 0.8rem;border-radius:8px;border-left:4px solid #667eea;font-size:0.8rem;color:#666;margin-bottom:1rem}
</style>"""

for key in ["chat_history","vectorstore","summary","total_chunks","client","all_chunks","dark_mode","answer_lang","use_pinecone"]:
    if key not in st.session_state:
        if key == "chat_history": st.session_state[key] = []
        elif key == "all_chunks": st.session_state[key] = []
        elif key == "dark_mode": st.session_state[key] = False
        elif key == "answer_lang": st.session_state[key] = "English"
        elif key == "use_pinecone": st.session_state[key] = False
        elif key == "total_chunks": st.session_state[key] = 0
        else: st.session_state[key] = None

st.markdown(get_css(st.session_state.dark_mode), unsafe_allow_html=True)
st.markdown('<div class="main-header"><h1>📄 RAGify</h1><p>Intelligent Document Q&A — Powered by Groq + LLaMA 3</p></div>', unsafe_allow_html=True)

def extract_text_ocr(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        page_text = pytesseract.image_to_string(img)
        text += page_text + "\n"
    doc.close()
    return text

def extract_tables_from_pdf(pdf_path):
    tables_text = ""
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        tabs = page.find_tables()
        if tabs.tables:
            for tab in tabs.tables:
                df = tab.to_pandas()
                tables_text += "\nTable:\n" + df.to_string() + "\n"
    doc.close()
    return tables_text

def extract_text_from_url(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return text[:10000]
    except Exception as e:
        return "Error: " + str(e)

def process_documents(files=None, urls=None, use_ocr=False):
    all_chunks = []
    if files:
        for f in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
            try:
                if use_ocr:
                    text = extract_text_ocr(tmp_path)
                    tables = extract_tables_from_pdf(tmp_path)
                    full_text = text + "\n" + tables
                    doc = Document(page_content=full_text, metadata={"source": f.name})
                    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                    chunks = splitter.split_documents([doc])
                else:
                    loader = PyMuPDFLoader(tmp_path)
                    docs = loader.load()
                    tables = extract_tables_from_pdf(tmp_path)
                    if tables:
                        docs.append(Document(page_content=tables, metadata={"source": f.name + "_tables"}))
                    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                    chunks = splitter.split_documents(docs)
                all_chunks.extend(chunks)
            except Exception as e:
                st.error("Error: " + str(e))
            finally:
                os.unlink(tmp_path)
    if urls:
        for url in urls:
            if url.strip():
                text = extract_text_from_url(url)
                if not text.startswith("Error"):
                    doc = Document(page_content=text, metadata={"source": url})
                    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                    chunks = splitter.split_documents([doc])
                    all_chunks.extend(chunks)
                    st.success("✅ URL: " + url)
                else:
                    st.error(text)
    return all_chunks

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    dark = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    if dark != st.session_state.dark_mode:
        st.session_state.dark_mode = dark
        st.rerun()
    lang = st.selectbox("🌐 Answer Language", ["English", "Hindi", "Hinglish"])
    st.session_state.answer_lang = lang
    model = st.selectbox("🤖 LLM Model", [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "gemma2-9b-it",
        "mixtral-8x7b-32768"
    ])
    st.markdown("---")

    # Storage Selection
    st.markdown("## 💾 Storage")
    storage_type = st.radio("Vector Store", ["⚡ FAISS (Session)", "🌐 Pinecone (Persistent)"])
    use_pinecone = storage_type == "🌐 Pinecone (Persistent)"

    if use_pinecone and PINECONE_API_KEY:
        st.success("✅ Pinecone Connected!")
    elif use_pinecone:
        st.error("❌ Pinecone key missing!")

    st.markdown("---")
    groq_key = st.text_input("🔑 Groq API Key", type="password", placeholder="gsk_...")
    if groq_key:
        st.session_state.client = Groq(api_key=groq_key)
        st.success("✅ Connected!")
    else:
        st.warning("⚠️ Enter Groq API Key")

    st.markdown("---")
    st.markdown("## 📁 Upload Documents")
    input_type = st.radio("Input Type", ["📄 PDF Files", "🌐 URL", "📄 + 🌐 Both"])
    use_ocr = st.checkbox("🔍 Enable OCR (for scanned PDFs)")

    uploaded_files = None
    url_input = None

    if input_type in ["📄 PDF Files", "📄 + 🌐 Both"]:
        uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
    if input_type in ["🌐 URL", "📄 + 🌐 Both"]:
        url_input = st.text_area("Enter URLs (one per line):", placeholder="https://example.com")

    if groq_key and (uploaded_files or url_input):
        if st.button("🚀 Process", type="primary", use_container_width=True):
            with st.spinner("⏳ Processing..."):
                urls = url_input.strip().split("\n") if url_input else None
                all_chunks = process_documents(files=uploaded_files, urls=urls, use_ocr=use_ocr)
                if all_chunks:
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    if use_pinecone and PINECONE_API_KEY:
                        try:
                            pc = Pinecone(api_key=PINECONE_API_KEY)
                            if PINECONE_INDEX not in pc.list_indexes().names():
                                pc.create_index(
                                    name=PINECONE_INDEX,
                                    dimension=384,
                                    metric="cosine",
                                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                                )
                            vectorstore = PineconeVectorStore.from_documents(
                                all_chunks,
                                embeddings,
                                index_name=PINECONE_INDEX
                            )
                            st.success("✅ Saved to Pinecone!")
                        except Exception as e:
                            st.error("Pinecone error: " + str(e))
                            vectorstore = FAISS.from_documents(all_chunks, embeddings)
                    else:
                        vectorstore = FAISS.from_documents(all_chunks, embeddings)

                    st.session_state.vectorstore = vectorstore
                    st.session_state.all_chunks = all_chunks
                    st.session_state.total_chunks = len(all_chunks)
                    sample = " ".join([c.page_content for c in all_chunks[:5]])
                    resp = st.session_state.client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": "Summarize in 3 sentences:\n" + sample}],
                        max_tokens=150
                    )
                    st.session_state.summary = resp.choices[0].message.content
                    st.session_state.chat_history = []
                    st.success("✅ " + str(len(all_chunks)) + " chunks ready!")
                else:
                    st.error("No content found!")
            st.rerun()

    # Load from Pinecone
    if use_pinecone and PINECONE_API_KEY and not st.session_state.vectorstore:
        if st.button("📂 Load from Pinecone", use_container_width=True):
            with st.spinner("Loading from Pinecone..."):
                try:
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    vectorstore = PineconeVectorStore(
                        index_name=PINECONE_INDEX,
                        embedding=embeddings,
                        pinecone_api_key=PINECONE_API_KEY
                    )
                    st.session_state.vectorstore = vectorstore
                    st.success("✅ Loaded from Pinecone!")
                except Exception as e:
                    st.error("Error: " + str(e))
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
    with c1: st.info("**Step 1**\n\n🔑 Enter Groq API Key")
    with c2: st.info("**Step 2**\n\n📁 PDF ya 🌐 URL")
    with c3: st.info("**Step 3**\n\n💬 Ask questions!")
    st.markdown("---")
    st.markdown("### ✨ Pro Features")
    f1, f2, f3, f4, f5 = st.columns(5)
    with f1: st.success("🔍 Hybrid Search")
    with f2: st.success("🧠 Memory")
    with f3: st.success("🌙 Dark Mode")
    with f4: st.success("🎤 Voice Input")
    with f5: st.success("📋 Copy Answer")
    f6, f7, f8, f9, f10 = st.columns(5)
    with f6: st.success("📝 Quiz Gen")
    with f7: st.success("🃏 Flashcards")
    with f8: st.success("🤖 Multi-Model")
    with f9: st.success("🌐 Pinecone")
    with f10: st.success("🔍 OCR")
else:
    if st.session_state.summary:
        with st.expander("📋 Document Summary", expanded=False):
            st.write(st.session_state.summary)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 Chat", "📝 Quiz Generator", "🃏 Flashcards", "📊 Model Comparison", "🔬 Chunking Analysis"])

    with tab1:
        st.markdown("### 💬 Conversation")
        for item in st.session_state.chat_history:
            st.markdown('<div class="chat-user">❓ ' + item["question"] + '</div>', unsafe_allow_html=True)
            conf_color = "green" if item["confidence"] == "High" else "orange" if item["confidence"] == "Medium" else "red"
            st.markdown('<div class="chat-bot">🤖 ' + item["answer"] + '<br><small style="color:' + conf_color + '">Confidence: ' + item["confidence"] + ' | ' + item["time"] + '</small></div>', unsafe_allow_html=True)
            with st.expander("📋 Copy Answer"):
                st.code(item["answer"], language=None)
            st.markdown('<div class="source-box">📄 ' + item["source"] + '...</div>', unsafe_allow_html=True)

        st.markdown("---")
        col_voice, col_text, col_btn = st.columns([1, 4, 1])
        with col_voice:
            audio = mic_recorder(start_prompt="🎤 Record", stop_prompt="⏹️ Stop", key="mic")
        with col_text:
            question = st.text_input("", placeholder="Ask anything or use mic...", label_visibility="collapsed", key="text_input")
        with col_btn:
            ask = st.button("Ask 🔍", type="primary", use_container_width=True)

        voice_question = None
        if audio is not None:
            if groq_key:
                with st.spinner("🎤 Transcribing..."):
                    try:
                        audio_client = Groq(api_key=groq_key)
                        transcription = audio_client.audio.transcriptions.create(
                            file=("audio.wav", audio['bytes'], "audio/wav"),
                            model="whisper-large-v3",
                        )
                        voice_question = transcription.text
                        st.success("📝 Voice: " + voice_question)
                    except Exception as e:
                        st.error("Voice error: " + str(e))

        final_question = voice_question if voice_question else question

        if (ask or voice_question) and final_question and st.session_state.client:
            with st.spinner("🤖 Thinking..."):
                semantic_docs = st.session_state.vectorstore.similarity_search(final_question, k=3)
                if st.session_state.all_chunks:
                    question_words = set(final_question.lower().split())
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
                else:
                    docs = semantic_docs

                try:
                    score_docs = st.session_state.vectorstore.similarity_search_with_score(final_question, k=3)
                    scores = [s for _, s in score_docs]
                    avg = sum(scores) / len(scores)
                    confidence = "High" if avg < 0.3 else "Medium" if avg < 0.6 else "Low"
                except:
                    confidence = "Medium"

                memory_context = ""
                if st.session_state.chat_history:
                    for item in st.session_state.chat_history[-3:]:
                        memory_context += "Q: " + item["question"] + "\nA: " + item["answer"] + "\n\n"

                context = "\n\n".join([d.page_content for d in docs])
                lang_instruction = ""
                if st.session_state.answer_lang == "Hindi":
                    lang_instruction = "Answer in Hindi only. "
                elif st.session_state.answer_lang == "Hinglish":
                    lang_instruction = "Answer in Hinglish. "

                prompt = lang_instruction + "Previous:\n" + memory_context + "\nContext:\n" + context + "\n\nQuestion: " + final_question + "\nAnswer:"
                response = st.session_state.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "Answer based on document context only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=400
                )
                answer = response.choices[0].message.content
                st.session_state.chat_history.append({
                    "question": final_question,
                    "answer": answer,
                    "confidence": confidence,
                    "source": docs[0].page_content[:250],
                    "time": datetime.now().strftime("%H:%M")
                })
                st.rerun()

    with tab2:
        st.markdown("### 📝 Quiz Generator")
        num_q = st.slider("Number of Questions", 3, 10, 5)
        if st.button("🎯 Generate Quiz", type="primary"):
            with st.spinner("Generating..."):
                sample = " ".join([c.page_content for c in st.session_state.all_chunks[:10]]) if st.session_state.all_chunks else ""
                if not sample:
                    docs = st.session_state.vectorstore.similarity_search("main topic", k=5)
                    sample = " ".join([d.page_content for d in docs])
                resp = st.session_state.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Generate " + str(num_q) + " MCQ with 4 options and answer. Format: Q1. [q]\nA) B) C) D)\nAnswer: [x]\n\nText: " + sample[:2000]}],
                    max_tokens=1000
                )
                quiz = resp.choices[0].message.content
                st.text_area("Quiz", quiz, height=400)
                st.download_button("📥 Download", quiz, "quiz.txt")

    with tab3:
        st.markdown("### 🃏 Flashcard Generator")
        num_f = st.slider("Number of Flashcards", 5, 20, 10)
        if st.button("🃏 Generate", type="primary"):
            with st.spinner("Generating..."):
                sample = " ".join([c.page_content for c in st.session_state.all_chunks[:10]]) if st.session_state.all_chunks else ""
                if not sample:
                    docs = st.session_state.vectorstore.similarity_search("main topic", k=5)
                    sample = " ".join([d.page_content for d in docs])
                resp = st.session_state.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Generate " + str(num_f) + " flashcards. Format:\nFront: [concept]\nBack: [definition]\n---\n\nText: " + sample[:2000]}],
                    max_tokens=1000
                )
                flashcards_text = resp.choices[0].message.content
                cards = flashcards_text.split("---")
                for i, card in enumerate(cards[:num_f]):
                    if "Front:" in card and "Back:" in card:
                        parts = card.strip().split("Back:")
                        front = parts[0].replace("Front:", "").strip()
                        back = parts[1].strip() if len(parts) > 1 else ""
                        with st.expander("Card " + str(i+1) + ": " + front[:50]):
                            st.info("**Front:** " + front)
                            st.success("**Back:** " + back)
                st.download_button("📥 Download", flashcards_text, "flashcards.txt")

    with tab4:
        st.markdown("### 📊 Model Comparison")
        compare_q = st.text_input("Question:", placeholder="What is binary search?")
        models_to_compare = st.multiselect(
            "Select models:",
            ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "gemma2-9b-it", "mixtral-8x7b-32768"],
            default=["llama-3.1-8b-instant", "gemma2-9b-it"]
        )
        if st.button("🔄 Compare", type="primary") and compare_q and models_to_compare:
            docs = st.session_state.vectorstore.similarity_search(compare_q, k=2)
            context = "\n".join([d.page_content for d in docs])
            prompt = "Answer briefly:\n" + context + "\n\nQ: " + compare_q + "\nA:"
            cols = st.columns(len(models_to_compare))
            for i, m in enumerate(models_to_compare):
                with cols[i]:
                    with st.spinner(m):
                        try:
                            resp = st.session_state.client.chat.completions.create(
                                model=m,
                                messages=[{"role": "user", "content": prompt}],
                                max_tokens=200
                            )
                            st.markdown("**" + m + "**")
                            st.success(resp.choices[0].message.content)
                        except Exception as e:
                            st.error(m + ": " + str(e))
    with tab5:
        st.markdown("### 🔬 Chunking Strategy Comparison")
        st.write("Compare different chunking strategies and their impact on retrieval quality!")

        compare_question = st.text_input(
            "Test Question:",
            placeholder="What is binary search?",
            key="chunk_compare_q"
        )

        if st.button("🔬 Run Chunking Analysis", type="primary"):
            if not compare_question:
                st.warning("Please enter a test question!")
            elif not st.session_state.all_chunks:
                st.warning("Please upload and process a PDF first!")
            else:
                with st.spinner("Running analysis on 3 strategies..."):
                    sample_text = " ".join([c.page_content for c in st.session_state.all_chunks[:20]])

                    strategies = [
                        {"name": "Small Chunks", "size": 200, "overlap": 20, "emoji": "🔹"},
                        {"name": "Medium Chunks", "size": 500, "overlap": 50, "emoji": "🔷"},
                        {"name": "Large Chunks", "size": 1000, "overlap": 100, "emoji": "🔵"},
                    ]

                    results = []
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

                    for strategy in strategies:
                        from langchain_core.documents import Document
                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=strategy["size"],
                            chunk_overlap=strategy["overlap"]
                        )
                        doc = Document(page_content=sample_text, metadata={"source": "test"})
                        chunks = splitter.split_documents([doc])
                        vs = FAISS.from_documents(chunks, embeddings)
                        retrieved = vs.similarity_search(compare_question, k=3)
                        context = "\n".join([d.page_content for d in retrieved])
                        resp = st.session_state.client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": "Answer briefly:\n" + context + "\n\nQ: " + compare_question + "\nA:"}],
                            max_tokens=150
                        )
                        answer = resp.choices[0].message.content
                        score_docs = vs.similarity_search_with_score(compare_question, k=3)
                        scores = [s for _, s in score_docs]
                        avg_score = sum(scores) / len(scores)
                        retrieval_quality = round((1 - min(avg_score, 1)) * 100, 1)

                        results.append({
                            "strategy": strategy["emoji"] + " " + strategy["name"],
                            "chunk_size": strategy["size"],
                            "num_chunks": len(chunks),
                            "answer": answer,
                            "retrieval_quality": retrieval_quality,
                            "avg_distance": round(avg_score, 3)
                        })

                st.markdown("### 📊 Results")
                col1, col2, col3 = st.columns(3)
                cols = [col1, col2, col3]

                best_quality = max(results, key=lambda x: x["retrieval_quality"])

                for i, result in enumerate(results):
                    with cols[i]:
                        is_best = result["strategy"] == best_quality["strategy"]
                        if is_best:
                            st.success("🏆 BEST STRATEGY")
                        st.markdown("**" + result["strategy"] + "**")
                        st.metric("Chunk Size", str(result["chunk_size"]) + " chars")
                        st.metric("Total Chunks", result["num_chunks"])
                        st.metric("Retrieval Quality", str(result["retrieval_quality"]) + "%")
                        st.metric("Avg Distance", result["avg_distance"])
                        with st.expander("📝 Answer"):
                            st.write(result["answer"])

                st.markdown("---")
                st.markdown("### 📈 Summary Table")

                import pandas as pd
                df = pd.DataFrame([{
                    "Strategy": r["strategy"],
                    "Chunk Size": r["chunk_size"],
                    "Total Chunks": r["num_chunks"],
                    "Retrieval Quality %": r["retrieval_quality"],
                    "Avg Distance": r["avg_distance"]
                } for r in results])
                st.dataframe(df, use_container_width=True)

                best = max(results, key=lambda x: x["retrieval_quality"])
                st.success("🏆 Best Strategy: **" + best["strategy"] + "** with " + str(best["retrieval_quality"]) + "% retrieval quality!")
                st.info("💡 Research Insight: " + best["strategy"] + " performs best for this document type. This finding can be included in your research paper!")
