from datetime import date

import numpy as np
import streamlit as st
from fastembed import TextEmbedding
from groq import Groq
from pypdf import PdfReader

APP_NAME = 'NoteGenie'
CHUNK_SIZE = 800
OVERLAP = 150
TOP_K = 4
MAX_PAGES = 40
MAX_FILE_MB = 5
MAX_QUESTIONS = 15   # per visitor session
DAILY_LIMIT = 150    # all visitors together, protects the free API quota

st.set_page_config(
    page_title='NoteGenie | AI Study Assistant',
    page_icon='📘',
    layout='wide',
    initial_sidebar_state='expanded'
)


def secret(name, default=None):
    try:
        return st.secrets[name]
    except Exception:
        return default


MODEL = secret('GROQ_MODEL', 'openai/gpt-oss-20b')

SYSTEM_PROMPT = (
    'You are NoteGenie, a helpful study assistant. '
    "Answer clearly and accurately in the user's language. "
    'Do not invent information. If you are unsure, say so.'
)

MCQ_PROMPT = (
    'Create 5 multiple-choice questions from the TEXT below. Rules: '
    'every question must be answerable using ONLY the text; give 4 options '
    '(A, B, C, D) with exactly one correct option; after each question write '
    "'Answer:' with the letter, then 'Evidence:' with the exact sentence from "
    'the text that proves it. If the text does not have enough facts, create '
    'fewer questions. Do not use outside knowledge. TEXT: '
)

SUMMARY_PROMPT = (
    'Write concise study notes as bullet points that summarize the TEXT below. '
    'Use only the text. Do not add outside information. TEXT: '
)

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

if not secret('GROQ_API_KEY'):
    st.error('GROQ_API_KEY is missing. Add it in the app secrets.')
    st.stop()

defaults = {
    'messages': [], 'texts': [], 'metas': [], 'vectors': None,
    'files': [], 'used': 0, 'use_documents': True,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


@st.cache_resource
def load_embedder():
    return TextEmbedding()


@st.cache_resource
def usage():
    return {'day': date.today().isoformat(), 'count': 0}


def embed(texts):
    vectors = np.array(list(load_embedder().embed(texts)), dtype='float32')
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-9)


def split_text(text):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - OVERLAP
    return chunks


def index_pdf(file):
    reader = PdfReader(file)
    texts, metas = [], []
    for page_no, page in enumerate(reader.pages[:MAX_PAGES], start=1):
        text = page.extract_text() or ''
        for chunk in split_text(text):
            if chunk.strip():
                texts.append(chunk)
                metas.append({'source': file.name, 'page': page_no})
    if not texts:
        return 0
    vectors = embed(texts)
    st.session_state.texts += texts
    st.session_state.metas += metas
    if st.session_state.vectors is None:
        st.session_state.vectors = vectors
    else:
        st.session_state.vectors = np.vstack([st.session_state.vectors, vectors])
    st.session_state.files.append(file.name)
    return len(texts)


def as_source(i):
    meta = st.session_state.metas[i]
    return {'text': st.session_state.texts[i], 'source': meta['source'], 'page': meta['page']}


def search_notes(question):
    if st.session_state.vectors is None:
        return []
    scores = st.session_state.vectors @ embed([question])[0]
    top = np.argsort(-scores)[:TOP_K]
    return [as_source(int(i)) for i in top]


def sample_file_text(name, max_chunks=6):
    idx = [i for i, m in enumerate(st.session_state.metas) if m['source'] == name]
    if not idx:
        return [], ''
    step = max(1, len(idx) // max_chunks)
    picked = idx[::step][:max_chunks]
    return [as_source(i) for i in picked], ' '.join(st.session_state.texts[i] for i in picked)


def allow_request():
    record = usage()
    today = date.today().isoformat()
    if record['day'] != today:
        record['day'] = today
        record['count'] = 0
    if record['count'] >= DAILY_LIMIT:
        return False, 'The free daily limit of this demo has been reached. Please try again tomorrow.'
    if st.session_state.used >= MAX_QUESTIONS:
        return False, f'You have used all {MAX_QUESTIONS} questions for this session. Please come back later.'
    record['count'] += 1
    st.session_state.used += 1
    return True, ''


def stream_answer(messages):
    client = Groq(api_key=secret('GROQ_API_KEY'))
    got_text = False
    stream = client.chat.completions.create(
        model=MODEL, messages=messages, stream=True,
        temperature=0.2, max_completion_tokens=2500,
        extra_body={'reasoning_effort': 'low'}
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        text = chunk.choices[0].delta.content
        if text:
            got_text = True
            yield text
    if not got_text:
        yield 'The model returned an empty answer. Please try again or rephrase the question.'



def show_sources(sources):
    with st.expander('Sources'):
        for s in sources:
            st.caption(f"{s['source']} - page {s['page']}")
            st.write(s['text'])


pending = None

with st.sidebar:
    st.markdown('# 📘 NoteGenie')
    st.caption('YOUR ONLINE STUDY COMPANION')
    st.divider()

    st.markdown('### 📚 Knowledge Base')
    st.warning('Do not upload private or confidential files. Text from your PDF '
               'is sent to an external AI service to generate answers.')
    uploaded = st.file_uploader('Add PDF documents', type=['pdf'], accept_multiple_files=True)

    if st.button('＋ Index uploaded PDFs', use_container_width=True):
        if not uploaded:
            st.warning('Please upload at least one PDF.')
        for f in uploaded or []:
            if f.name in st.session_state.files:
                st.info(f'{f.name} is already indexed.')
            elif f.size > MAX_FILE_MB * 1024 * 1024:
                st.error(f'{f.name} is larger than {MAX_FILE_MB} MB.')
            else:
                with st.spinner(f'Reading {f.name}...'):
                    try:
                        n = index_pdf(f)
                        if n:
                            st.success(f'{f.name}: {n} chunks indexed.')
                        else:
                            st.warning(f'No readable text in {f.name}. It may be a scanned PDF.')
                    except Exception as error:
                        st.error(f'Could not read {f.name}: {error}')

    st.markdown(f"**Indexed documents:** {len(st.session_state.files)}")
    st.markdown(f"**Indexed chunks:** {len(st.session_state.texts)}")
    st.toggle('Answer from my documents', key='use_documents')

    st.divider()
    st.markdown('### 🎓 Study Tools')
    if st.session_state.files:
        chosen = st.selectbox('Choose a document', st.session_state.files)
        col_a, col_b = st.columns(2)
        if col_a.button('📝 Summarize', use_container_width=True):
            srcs, text = sample_file_text(chosen)
            pending = {'display': f'Summarize: {chosen}', 'prompt': SUMMARY_PROMPT + text,
                       'sources': srcs, 'use_history': False}
        if col_b.button('❓ MCQs', use_container_width=True):
            srcs, text = sample_file_text(chosen)
            pending = {'display': f'Generate MCQs: {chosen}', 'prompt': MCQ_PROMPT + text,
                       'sources': srcs, 'use_history': False}
    else:
        st.caption('Index a PDF to unlock study tools.')

    st.divider()
    st.markdown('### ⚙️ Model Settings')
    st.markdown(f'**Model:** `{MODEL}`')
    st.markdown('**Mode:** 🌐 Online')
    st.caption(f"Questions left in this session: {MAX_QUESTIONS - st.session_state.used}")

    if st.button('🗑️ Clear conversation', use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if st.button('Clear my documents', use_container_width=True):
        st.session_state.texts = []
        st.session_state.metas = []
        st.session_state.vectors = None
        st.session_state.files = []
        st.rerun()

    st.divider()
    st.caption('NoteGenie · AI study assistant')

st.markdown("<h1 style='margin-bottom:0;'>✨ NoteGenie</h1>", unsafe_allow_html=True)
st.markdown('Your personal AI companion for learning, research and study.')
st.markdown(
    '<span style="background:#123b37;color:#78f0c2;padding:6px 12px;'
    'border-radius:20px;font-size:13px;">● ONLINE · RAG</span>',
    unsafe_allow_html=True
)
st.write('')

question = st.chat_input('Message NoteGenie...')

if question:
    sources = search_notes(question) if st.session_state.use_documents else []
    if sources:
        context = ' '.join(s['text'] for s in sources)
        prompt = (
            'Answer the question using ONLY the context below. '
            'If the answer is not in the context, say: I do not know from the documents. '
            f'CONTEXT: {context} QUESTION: {question} ANSWER:'
        )
    else:
        prompt = question
    pending = {'display': question, 'prompt': prompt, 'sources': sources, 'use_history': True}

if not st.session_state.messages and not pending:
    st.markdown('### 👋 Hello! What would you like to learn today?')
    st.write('Upload a PDF in the sidebar, then ask a question.')

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        if message.get('sources'):
            show_sources(message['sources'])

if pending:
    ok, reason = allow_request()
    if not ok:
        st.warning(reason)
    else:
        history = []
        if pending['use_history']:
            history = [{'role': m['role'], 'content': m['content']}
                       for m in st.session_state.messages[-4:]]

        st.session_state.messages.append({'role': 'user', 'content': pending['display']})
        with st.chat_message('user'):
            st.markdown(pending['display'])

        chat_messages = (
            [{'role': 'system', 'content': SYSTEM_PROMPT}]
            + history
            + [{'role': 'user', 'content': pending['prompt']}]
        )

        with st.chat_message('assistant'):
            try:
                answer = st.write_stream(stream_answer(chat_messages))
                if pending['sources']:
                    show_sources(pending['sources'])
            except Exception as error:
                answer = ('The AI service is busy or its free limit is reached. '
                          f'Please try again in a minute.\n\nTechnical details: `{error}`')
                st.error(answer)

        st.session_state.messages.append(
            {'role': 'assistant', 'content': answer, 'sources': pending['sources']})
