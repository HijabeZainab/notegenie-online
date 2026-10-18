import streamlit as st
import chromadb
import ollama
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

MODEL = "llama3.2"
CHUNK_SIZE = 800
OVERLAP = 150
TOP_K = 4

SYSTEM_PROMPT = (
    "You are NoteGenie, a helpful offline study assistant. "
    "Answer clearly and accurately in the user's language. "
    "Do not invent information. If you are unsure, say so."
)

MCQ_PROMPT = (
    "Create 5 multiple-choice questions from the TEXT below. Rules: "
    "every question must be answerable using ONLY the text; give 4 options "
    "(A, B, C, D) with exactly one correct option; after each question write "
    "'Answer:' with the letter, then 'Evidence:' with the exact sentence from "
    "the text that proves it. If the text does not have enough facts, create "
    "fewer questions. Do not use outside knowledge. TEXT: "
)

SUMMARY_PROMPT = (
    "Write concise study notes as bullet points that summarize the TEXT below. "
    "Use only the text. Do not add outside information. TEXT: "
)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="NoteGenie | Offline AI",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
.stApp { background: #0b1120; color: #e5edf8; }
[data-testid="stSidebar"] { background: #111b2e; border-right: 1px solid #263751; }
h1, h2, h3 { color: #eaf6ff !important; }
p, label, span { color: #d4dfef; }
div.stButton > button {
    background: #102b43; color: #5ce1e6; border: 1px solid #1e6074;
    border-radius: 10px; padding: 8px 15px;
}
div.stButton > button:hover { background: #16445a; border-color: #5ce1e6; }
[data-testid="stChatMessage"] {
    background: #111c30; border: 1px solid #243650;
    border-radius: 14px; padding: 14px;
}
[data-testid="stChatInput"] { border: 1px solid #2c6279; border-radius: 12px; }
div[data-testid="stFileUploader"] {
    background: #111c30; border: 1px dashed #2c6279;
    border-radius: 12px; padding: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "use_documents" not in st.session_state:
    st.session_state.use_documents = True

# ---------------- RESOURCES ----------------
@st.cache_resource
def load_resources():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="db")
    return embedder, client

embedder, client = load_resources()

def get_collection():
    return client.get_or_create_collection("notes")

# ---------------- HELPERS ----------------
def split_text(text):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - OVERLAP
    return chunks

def index_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    ids, texts, metas = [], [], []
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for j, chunk in enumerate(split_text(text)):
            if chunk.strip():
                ids.append(f"{uploaded_file.name}-{page_no}-{j}")
                texts.append(chunk)
                metas.append({"source": uploaded_file.name, "page": page_no, "chunk": j})
    if texts:
        vectors = embedder.encode(texts).tolist()
        get_collection().upsert(ids=ids, documents=texts, embeddings=vectors, metadatas=metas)
    return len(texts)

def list_files():
    data = get_collection().get(include=["metadatas"])
    return sorted({m["source"] for m in data["metadatas"]})

def search_notes(question):
    collection = get_collection()
    total = collection.count()
    if total == 0:
        return []
    vector = embedder.encode([question]).tolist()
    res = collection.query(query_embeddings=vector, n_results=min(TOP_K, total))
    return [
        {"text": d, "source": m["source"], "page": m["page"]}
        for d, m in zip(res["documents"][0], res["metadatas"][0])
    ]

def sample_file_text(name, max_chunks=6):
    data = get_collection().get(where={"source": name}, include=["documents", "metadatas"])
    items = sorted(
        zip(data["documents"], data["metadatas"]),
        key=lambda x: (x[1]["page"], x[1].get("chunk", 0))
    )
    if not items:
        return [], ""
    step = max(1, len(items) // max_chunks)
    picked = items[::step][:max_chunks]
    sources = [{"text": d, "source": m["source"], "page": m["page"]} for d, m in picked]
    return sources, " ".join(d for d, _ in picked)

def stream_answer(messages):
    for chunk in ollama.chat(model=MODEL, messages=messages, stream=True):
        yield chunk["message"]["content"]

def show_sources(sources):
    with st.expander("Sources"):
        for s in sources:
            st.caption(f"{s['source']} - page {s['page']}")
            st.write(s["text"])

# ---------------- SIDEBAR ----------------
pending = None

with st.sidebar:
    st.markdown("# 📘 NoteGenie")
    st.caption("YOUR PERSONAL OFFLINE STUDY COMPANION")
    st.divider()

    st.markdown("### 📚 Knowledge Base")
    st.write("Upload PDF files to ask questions about their content.")

    uploaded_files = st.file_uploader(
        "Add PDF documents", type=["pdf"], accept_multiple_files=True
    )

    if st.button("＋ Index uploaded PDFs", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
        else:
            total_new = 0
            for f in uploaded_files:
                with st.spinner(f"Reading {f.name}..."):
                    try:
                        total_new += index_pdf(f)
                    except Exception as error:
                        st.error(f"Could not read {f.name}: {error}")
            if total_new:
                st.success(f"Indexed {total_new} chunks.")
            else:
                st.warning("No readable text found. The PDFs may be scanned images.")

    files = list_files()
    st.markdown(f"**Indexed documents:** {len(files)}")
    st.markdown(f"**Indexed chunks:** {get_collection().count()}")

    st.toggle("Answer from my documents", key="use_documents")

    st.divider()
    st.markdown("### 🎓 Study Tools")
    if files:
        chosen = st.selectbox("Choose a document", files)
        col_a, col_b = st.columns(2)
        if col_a.button("📝 Summarize", use_container_width=True):
            srcs, text = sample_file_text(chosen)
            pending = {
                "display": f"Summarize: {chosen}",
                "prompt": SUMMARY_PROMPT + text,
                "sources": srcs,
                "use_history": False,
            }
        if col_b.button("❓ MCQs", use_container_width=True):
            srcs, text = sample_file_text(chosen)
            pending = {
                "display": f"Generate MCQs: {chosen}",
                "prompt": MCQ_PROMPT + text,
                "sources": srcs,
                "use_history": False,
            }
    else:
        st.caption("Index a PDF to unlock study tools.")

    st.divider()
    st.markdown("### ⚙️ Model Settings")
    st.markdown("**Model:** `llama3.2`")
    st.markdown("**Mode:** 🟢 Local / Offline")
    st.caption("The model must already be installed through Ollama.")

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("Clear all documents", use_container_width=True):
        client.delete_collection("notes")
        st.rerun()

    st.divider()
    st.caption("NoteGenie · Private local study assistant")

# ---------------- MAIN CHAT AREA ----------------
st.markdown("<h1 style='margin-bottom:0;'>✨ NoteGenie</h1>", unsafe_allow_html=True)
st.markdown("Your personal AI companion for learning, research and study.")
st.markdown(
    '<span style="background:#123b37;color:#78f0c2;'
    'padding:6px 12px;border-radius:20px;font-size:13px;">'
    '● LOCAL MODEL · LLAMA 3.2</span>',
    unsafe_allow_html=True
)
st.write("")

question = st.chat_input("Message NoteGenie...")

if question:
    sources = search_notes(question) if st.session_state.use_documents else []
    if sources:
        context = " ".join(s["text"] for s in sources)
        prompt = (
            "Answer the question using ONLY the context below. "
            "If the answer is not in the context, say: I do not know from the documents. "
            f"CONTEXT: {context} QUESTION: {question} ANSWER:"
        )
    else:
        prompt = question
    pending = {
        "display": question,
        "prompt": prompt,
        "sources": sources,
        "use_history": True,
    }

if not st.session_state.messages and not pending:
    st.markdown("### 👋 Hello! What would you like to learn today?")
    st.write("Ask a question, or upload a PDF in the sidebar.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 📖 Understand")
        st.caption("Get simple explanations of difficult concepts.")
    with col2:
        st.markdown("#### 📝 Summarize")
        st.caption("Turn study material into concise notes.")
    with col3:
        st.markdown("#### 💡 Explore")
        st.caption("Ask questions and explore new ideas.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            show_sources(message["sources"])

if pending:
    history = []
    if pending["use_history"]:
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[-4:]
        ]

    st.session_state.messages.append({"role": "user", "content": pending["display"]})
    with st.chat_message("user"):
        st.markdown(pending["display"])

    chat_messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": pending["prompt"]}]
    )

    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(stream_answer(chat_messages))
            if pending["sources"]:
                show_sources(pending["sources"])
        except Exception as error:
            answer = (
                "I couldn't connect to the local Ollama model.\n\n"
                "Please check that Ollama is installed, running, "
                "and that the llama3.2 model is available.\n\n"
                f"Technical details: `{error}`"
            )
            st.error(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": pending["sources"]}
    )