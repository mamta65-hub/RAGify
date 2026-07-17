import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
from datetime import datetime
import tempfile
import os
import random
from streamlit_mic_recorder import mic_recorder
import base64

st.set_page_config(page_title="RAGify", page_icon="📄", layout="wide")

# Dark Mode CSS
def get_css(dark_mode):
    if dark_mode:
        return """
<style>
.stApp {background:#1a1a2e;color:#e0e0e0}
.stSidebar {background:#16213e}
.main-header {background:linear-gradient(135deg,#667eea,#764ba2);padding:2rem;border-radius:15px;text-align:center;color:white;margin-bottom:2rem}
.chat-user {background:#667eea;color:white;padding:0.8rem 1.2rem;border-radius:18px 18px 4px 18px;margin:0.5rem 0;max-width:80%;margin-left:auto;text-align:right}
.chat-bot {background:#2d2d44;color:#e0e0e0;padding:0.8rem 1.2rem;border-radius:18px 18px 18px 4px;margin:0.5rem 0;max-width:80%;border:1px solid #444}
.source-box {background:#252540;padding:0.5rem 0.8rem;border-radius:8px;border-left:4px solid #667eea;font-size:0.8rem;color:#aaa;margin-bottom:1rem}
.feature-card {background:#2d2d44;padding:1rem;border-radius:10px;text-align:center;margin:0.5rem}
</style>"""
    else:
        return """
<style>
.stApp {background:#f8f9fa}
.main-header {background:linear-gradient(135deg,#667eea,#764ba2);padding:2rem;border-radius:15px;text-align:center;color:white;margin-bottom:2rem}
.chat-user {background:#667eea;color:white;padding:0.8rem 1.2rem;border-radius:18px 18px 4px 18px;margin:0.5rem 0;max-width:80%;margin-left:auto;text-align:right}
.chat-bot {background:white;color:#333;padding:0.8rem 1.2rem;border-radius:18px 18px 18px 4px;margin:0.5rem 0;max-width:80%;border:1px solid #e0e0e0;box-shadow:0 2px 5px rgba(0,0,0,0.1)}
.source-box {background:#f8f9fa;padding:0.5rem 0.8rem;border-radius:8px;border-left:4px solid #667eea;font-size:0.8rem;color:#666;margin-bottom:1rem}
</style>"""

# Session State Init
for key in ["chat_history","vectorstore","summary","total_chunks","client","all_chunks","dark_mode","answer_lang","active_tab"]:
    if key not in st.session_state:
        if key == "chat_history": st.session_state[key] = []
        elif key == "all_chunks": st.session_state[key] = []
        elif key == "dark_mode": st.session_state[key] = False
        elif key == "answer_lang": st.session_state[key] = "English"
        elif key == "active_tab": st.session_state[key] = "chat"
        elif key == "total_chunks": st.session_state[key] = 0
        else: st.session_state[key] = None

st.markdown(get_css(st.session_state.dark_mode), unsafe_allow_html=True)
st.markdown('<div class="main-header"><h1>📄 RAGify</h1><p>Intelligent Document Q&A — Powered by Groq + LLaMA 3</p></div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    # Dark Mode Toggle
    dark = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    if dark != st.session_state.dark_mode:
        st.session_state.dark_mode = dark
        st.rerun()

    # Language Selection
    lang = st.selectbox("🌐 Answer Language", ["English", "Hindi", "Hinglish"])
    st.session_state.answer_lang = lang

    # Model Selection
    model = st.selectbox("🤖 LLM Model", [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "gemma2-9b-it",
        "mixtral-8x7b-32768"
    ])

    st.markdown("---")
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
                    try:
                        loader = PyMuPDFLoader(tmp_path)
                        docs = loader.load()
                        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                        chunks = splitter.split_documents(docs)
                        all_chunks.extend(chunks)
                    except Exception as e:
                        st.error("Error: " + str(e))
                    finally:
                        os.unlink(tmp_path)

                if all_chunks:
                    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
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
                    st.error("No text found! Try another PDF.")
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

# Main Area
if not st.session_state.vectorstore:
    c1, c2, c3 = st.columns(3)
    with c1: st.info("**Step 1**\n\n🔑 Enter Groq API Key")
    with c2: st.info("**Step 2**\n\n📁 Upload PDF files")
    with c3: st.info("**Step 3**\n\n💬 Ask questions!")
    st.markdown("---")
    st.markdown("### ✨ Pro Features")
    f1, f2, f3, f4, f5 = st.columns(5)
    with f1: st.success("🔍 Hybrid Search")
    with f2: st.success("🧠 Memory")
    with f3: st.success("📊 Confidence")
    with f4: st.success("🌙 Dark Mode")
    with f5: st.success("🌐 Multi-Lang")
    f6, f7, f8, f9, f10 = st.columns(5)
    with f6: st.success("📝 Quiz Gen")
    with f7: st.success("🃏 Flashcards")
    with f8: st.success("🤖 Multi-Model")
    with f9: st.success("📥 Export Chat")
    with f10: st.success("💬 Chat History")

else:
    if st.session_state.summary:
        with st.expander("📋 Document Summary", expanded=False):
            st.write(st.session_state.summary)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📝 Quiz Generator", "🃏 Flashcards", "📊 Model Comparison"])

    # ── TAB 1: CHAT ──
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
            audio = mic_recorder(
                start_prompt="🎤",
                stop_prompt="⏹️",
                key="mic"
            )

        with col_text:
            question = st.text_input("", placeholder="Ask anything or use mic...", label_visibility="collapsed")

        with col_btn:
            ask = st.button("Ask 🔍", type="primary", use_container_width=True)

        if audio:
            st.info("🎤 Voice recorded! Processing...")
            audio_client = Groq(api_key=groq_key)
            transcription = audio_client.audio.transcriptions.create(
                file=("audio.wav", audio['bytes'], "audio/wav"),
                model="whisper-large-v3",
            )
            question = transcription.text
            st.success("Transcribed: " + question)

        if ask and question and st.session_state.client:

    # ── TAB 2: QUIZ GENERATOR ──
    with tab2:
        st.markdown("### 📝 Quiz Generator")
        st.write("Generate MCQ questions from your document!")
        num_q = st.slider("Number of Questions", 3, 10, 5)
        if st.button("🎯 Generate Quiz", type="primary"):
            with st.spinner("Generating quiz..."):
                sample = " ".join([c.page_content for c in st.session_state.all_chunks[:10]])
                resp = st.session_state.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Generate " + str(num_q) + " MCQ questions with 4 options (A,B,C,D) and correct answer from this text. Format: Q1. [question]\nA) [option]\nB) [option]\nC) [option]\nD) [option]\nAnswer: [correct]\n\nText: " + sample[:2000]}],
                    max_tokens=1000
                )
                quiz = resp.choices[0].message.content
                st.markdown("### 📋 Generated Quiz")
                st.text_area("Quiz Questions", quiz, height=400)
                st.download_button("📥 Download Quiz", quiz, "ragify_quiz.txt")

    # ── TAB 3: FLASHCARDS ──
    with tab3:
        st.markdown("### 🃏 Flashcard Generator")
        st.write("Generate study flashcards from your document!")
        num_f = st.slider("Number of Flashcards", 5, 20, 10)
        if st.button("🃏 Generate Flashcards", type="primary"):
            with st.spinner("Generating flashcards..."):
                sample = " ".join([c.page_content for c in st.session_state.all_chunks[:10]])
                resp = st.session_state.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Generate " + str(num_f) + " flashcards from this text. Format each as:\nFront: [concept/term]\nBack: [definition/explanation]\n---\n\nText: " + sample[:2000]}],
                    max_tokens=1000
                )
                flashcards_text = resp.choices[0].message.content
                cards = flashcards_text.split("---")
                st.markdown("### 🃏 Your Flashcards")
                for i, card in enumerate(cards[:num_f]):
                    if "Front:" in card and "Back:" in card:
                        parts = card.strip().split("Back:")
                        front = parts[0].replace("Front:", "").strip()
                        back = parts[1].strip() if len(parts) > 1 else ""
                        with st.expander("Card " + str(i+1) + ": " + front[:50]):
                            st.info("**Front:** " + front)
                            st.success("**Back:** " + back)
                st.download_button("📥 Download Flashcards", flashcards_text, "ragify_flashcards.txt")

    # ── TAB 4: MODEL COMPARISON ──
    with tab4:
        st.markdown("### 📊 Model Comparison")
        st.write("Compare different AI models on the same question!")
        compare_q = st.text_input("Enter question to compare:", placeholder="What is binary search?")
        models_to_compare = st.multiselect(
            "Select models to compare:",
            ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "gemma2-9b-it", "mixtral-8x7b-32768"],
            default=["llama-3.1-8b-instant", "gemma2-9b-it"]
        )
        if st.button("🔄 Compare Models", type="primary") and compare_q and models_to_compare:
            docs = st.session_state.vectorstore.similarity_search(compare_q, k=2)
            context = "\n".join([d.page_content for d in docs])
            prompt = "Answer briefly using this context:\n" + context + "\n\nQuestion: " + compare_q + "\nAnswer:"
            cols = st.columns(len(models_to_compare))
            for i, m in enumerate(models_to_compare):
                with cols[i]:
                    with st.spinner(m + "..."):
                        try:
                            resp = st.session_state.client.chat.completions.create(
                                model=m,
                                messages=[{"role": "user", "content": prompt}],
                                max_tokens=200
                            )
                            answer = resp.choices[0].message.content
                            st.markdown("**" + m + "**")
                            st.success(answer)
                        except Exception as e:
                            st.error(m + " failed: " + str(e))
