import streamlit as st
import os
from src.extractor import extract_text
from src.chunker import split_chunks
from src.vector_store import build_vector_store, search
from src.agent import agent_answer

# Page config
st.set_page_config(
    page_title="Vancity Research Agent",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stTextInput>div>div>input {
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📊 Vancity Research Agent</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered analysis of Vancity\'s yearly financial reports</p>', unsafe_allow_html=True)

# Initialize session state
if 'index' not in st.session_state:
    with st.spinner('🔄 Loading documents and building search index... (this takes ~30 seconds on first run)'):
        # Load documents
        docs = {}
        pdf_files = {
            "2024 Annual Report": "data/Vancity-2024-Annual-Report.pdf",
            "2024 Financial Statements": "data/2024-Consolidated-Financial-Statements.pdf",
            "2023 Annual Report": "data/2023-annual-report.pdf",
            "2023 Financial Statements": "data/2023-Consolidated-Financial-Statements.pdf",
            "2024 Climate Report": "data/Vancity-2024-Climate-Report.pdf",
            "2023 Climate Report": "data/2023-Climate-Report.pdf",
            "2024 Accountability Statements": "data/Vancity-2024-Accountability-Statements.pdf",
            "2023 Accountability Statements": "data/2023-Accountability-Statements.pdf"
        }

        all_text = ""
        for name, path in pdf_files.items():
            if os.path.exists(path):
                text = extract_text(path)
                all_text += f"\n\n=== {name} ===\n\n{text}"

        # Build index
        chunks = split_chunks(all_text)
        index, chunks = build_vector_store(chunks)

        st.session_state.index = index
        st.session_state.chunks = chunks
        st.success(f'✅ Ready! {len(chunks)} document chunks indexed.')

# Sidebar with example questions
with st.sidebar:
    st.header("📋 Example Questions")
    example_questions = [
        "What was Vancity's total revenue in 2024?",
        "Compare total assets between 2023 and 2024",
        "What are Vancity's climate commitments?",
        "How did net income change from 2023 to 2024?",
        "What is Vancity's approach to sustainability?"
    ]

    for q in example_questions:
        if st.button(q, key=q):
            st.session_state.question = q

    st.divider()
    st.caption("💡 Tip: Ask comparative questions across years for better insights!")

# Main chat interface
question = st.text_input(
    "Ask a question about Vancity's financial reports:",
    value=st.session_state.get('question', ''),
    placeholder="e.g., Compare Vancity's key financial metrics between 2023 and 2024",
    key="question_input"
)

if st.button("🔍 Search", type="primary", use_container_width=True) or (question and st.session_state.get('question') != st.session_state.get('last_question')):
    if question:
        st.session_state.last_question = question

        with st.spinner('🤔 Analyzing documents...'):
            result = agent_answer(
                question,
                st.session_state.index,
                st.session_state.chunks
            )

        # Display answer
        st.markdown("### 📝 Answer")
        st.markdown(result['answer'])

        # Display search steps in expander
        with st.expander("🔍 View search process"):
            for step in result['steps']:
                st.text(step)
            st.info(f"Total steps: {result['total_steps']}")

# Footer
st.divider()
st.caption("Powered by Claude AI • Data from Vancity's official reports")
