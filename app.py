import re
import streamlit as st
from main import initialize_backend
from src.agent import agent_answer


def render(text: str):
    """Render markdown, escaping bare $ so Streamlit doesn't treat them as LaTeX."""
    st.markdown(re.sub(r'\$(?=[\d(])', r'\\$', text))

st.set_page_config(
    page_title="Vancity Research Agent",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 0.5rem; }
    .sub-header { text-align: center; color: #666; margin-bottom: 1.5rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">📊 Vancity Research Agent</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">AI-powered analysis of Vancity\'s yearly financial reports</p>',
    unsafe_allow_html=True
)

# ── Initialize backend once ────────────────────────────────────────────────
if 'backend_loaded' not in st.session_state:
    with st.spinner('Loading documents and building search index… (first run ~30 s)'):
        index, chunks = initialize_backend()
        st.session_state.index = index
        st.session_state.chunks = chunks
        st.session_state.backend_loaded = True

# ── Chat history ───────────────────────────────────────────────────────────
if 'messages' not in st.session_state:
    st.session_state.messages = []

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Example Questions")
    examples = [
        "What was Vancity's total revenue in 2024?",
        "Compare total assets between 2023 and 2024",
        "How did net income change from 2023 to 2024?",
        "What are Vancity's climate commitments?",
        "What is Vancity's approach to sustainability?",
    ]
    for q in examples:
        if st.button(q, key=f"ex_{q}", use_container_width=True):
            st.session_state.pending_question = q

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("💡 Ask comparative questions across years for deeper insights.")

# ── Render existing chat history ───────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        render(msg['content'])

# ── Accept new input ───────────────────────────────────────────────────────
# Sidebar example buttons set pending_question; chat_input handles typed questions.
pending = st.session_state.pop('pending_question', None)
user_input = st.chat_input("Ask about Vancity's financial reports…")
question = user_input or pending

if question:
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        status = st.empty()
        answer_placeholder = st.empty()
        status.markdown("*Thinking…*")

        token_buffer = []
        state = {"searches": 0, "answer_started": False}

        def on_token(text):
            token_buffer.append(text)
            current = "".join(token_buffer)
            search_count = current.count("SEARCH:")

            if not state["answer_started"]:
                if search_count > state["searches"]:
                    state["searches"] = search_count
                    query = current.split("SEARCH:")[-1].strip().split("\n")[0][:60]
                    status.markdown(f"🔍 *Searching documents for: {query}…*")
                elif "ANSWER:" in current:
                    state["answer_started"] = True
                    status.markdown("✍️ *Writing answer…*")
                elif "SEARCH:" not in current:
                    # Conversational reply — stream directly
                    answer_placeholder.markdown(re.sub(r'\$(?=[\d(])', r'\\$', current) + "▌")
            else:
                after = current.split("ANSWER:", 1)[1].lstrip()
                answer_placeholder.markdown(re.sub(r'\$(?=[\d(])', r'\\$', after) + "▌")

        result = agent_answer(
            question,
            st.session_state.index,
            st.session_state.chunks,
            on_token=on_token,
            history=st.session_state.messages[:-1],
        )

        status.empty()
        answer_placeholder.empty()
        render(result['answer'])

    st.session_state.messages.append({
        "role": "assistant",
        "content": result['answer'],
    })

# ── Footer ─────────────────────────────────────────────────────────────────
st.divider()
st.caption("Powered by Claude AI • Data from Vancity's official reports")