# Vancity AI Research Assistant — AI Spec

## Why

Build a RAG + agentic pipeline that answers questions about Vancity's 2023 
and 2024 documents. Demonstrates practical GenAI engineering: embeddings, 
vector retrieval, agentic loops, prompt engineering, and responsible AI output.

## What

A working CLI + Streamlit chat agent that ingests 10 Vancity PDFs, indexes 
them with FAISS, and answers questions using a Claude agent loop with source 
citation, confidence scoring, streaming output, and within-session memory.

---

## Constraints

### Must
- Python 3.10+
- Anthropic SDK for Claude (model: `claude-sonnet-4-6`)
- sentence-transformers all-MiniLM-L6-v2 for embeddings
- faiss-cpu for vector storage
- pdfplumber for PDF extraction (table-aware)
- streamlit for frontend
- python-dotenv for API key management
- Save FAISS index to data/faiss.index after first build
- Stream Claude responses via on_token callback; escape `$` signs to prevent LaTeX rendering
- Every answer: Summary + Key Figures table + Source + Confidence signal
- Cap agent loop at 10 steps max; return last Claude output if max steps reached
- Pass conversation history to agent on each turn for within-session memory

### Must Not
- Do not rebuild index on every run — check index_exists() first
- Do not hardcode API key — use .env only
- Do not use outside knowledge — answers grounded in retrieved chunks only

### Out of Scope
- Authentication, multi-user, cloud deployment, fine-tuning
- Persistent memory across sessions (resets on page refresh)

---

## Project Structure

```
vancity-agent/
├── data/
│   ├── *.pdf                    (10 Vancity documents)
│   ├── faiss.index              (generated after first run)
│   └── chunks.pkl               (generated after first run)
├── src/
│   ├── __init__.py
│   ├── extractor.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── vector_store.py
│   └── agent.py
├── main.py
├── app.py
├── requirements.txt
└── .env
```

---

## Tasks

### T1: PDF Text Extraction
**What:** Extract raw text from all PDFs. Tables are converted to pipe-delimited 
rows so numbers stay associated with their headers. Non-table text is extracted 
separately to avoid duplication.

**Files:** `src/extractor.py`

**Key logic:**
```python
import pdfplumber

def extract_text(pdf_path: str) -> str:
    all_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            table_bboxes = []
            for table in tables:
                rows = [" | ".join(c.replace("\n"," ").strip() if c else "" for c in row)
                        for row in table if row]
                all_text += "\n".join(rows) + "\n\n"
                for t_obj in page.find_tables():
                    table_bboxes.append(t_obj.bbox)
            # Extract non-table text to avoid double-counting
            remaining = page
            for bbox in table_bboxes:
                remaining = remaining.outside_bbox(bbox)
            page_text = remaining.extract_text()
            if page_text:
                all_text += page_text + "\n"
    return all_text
```

**Why table-aware:** Plain `extract_text()` strips table structure, producing 
`28.4 28.8 28.3` with no year labels. Pipe-delimited format gives 
`Total assets | $28.4B | $28.8B | $28.3B` — Claude can cite figures accurately.

**Verify:** Output contains `Total assets | $28.4B` style rows for financial tables.

---

### T2: Text Chunking
**What:** Split extracted text into overlapping chunks for retrieval.

**Files:** `src/chunker.py`

**Key logic:**
```python
def split_chunks(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks
```

**Why 800/150:** Captures full paragraphs; 150 overlap = ~1-2 sentences safety net.

**Verify:** ~2200 chunks for all 10 docs combined.

---

### T3: Local Embeddings
**What:** Convert text to 384-dim vectors using local sentence-transformers.

**Files:** `src/embedder.py`

**Key logic:**
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')  # loads once at import time

def embed(text: str) -> list[float]:
    return model.encode(text).tolist()
```

**Tradeoff:** Local = free + private. Production would use Azure OpenAI embeddings.

**Verify:** `len(embed("test"))` returns 384.

---

### T4: Vector Store — Build, Save, Load, Search
**What:** FAISS index with disk persistence so we only embed once.

**Files:** `src/vector_store.py`

**Key logic:**
```python
INDEX_PATH  = "data/faiss.index"
CHUNKS_PATH = "data/chunks.pkl"

def index_exists() -> bool:
    return os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH)

def build_vector_store(chunks):
    # embed all chunks → numpy array → faiss.IndexFlatL2 → save to disk

def load_vector_store():
    # faiss.read_index + pickle.load → return (index, chunks)

def search(question, index, chunks, top_k=8) -> list[str]:
    # embed question → index.search → return top_k chunk texts
```

**Why top_k=8:** Broad comparison questions need more chunks to capture complete 
table rows split across chunk boundaries. 5 was too low for financial queries.

**Verify:**
- First run: builds and saves index files
- Second run: `index_exists()` returns True, loads instantly (~1s)

---

### T5: Claude Agent Loop
**What:** Agent that searches documents iteratively before answering, with 
streaming output, conversation memory, and a graceful fallback.

**Files:** `src/agent.py`

**Protocol:** Claude writes `SEARCH:` or `ANSWER:` — we parse and act on it.
If Claude responds without either token (e.g. conversational reply), treat it 
as the final answer immediately rather than cycling to max steps.

**Key logic:**
```python
def agent_answer(
    question: str,
    index,
    chunks: list[str],
    on_token=None,              # callback(str) for streaming — used by Streamlit
    history: list[dict] | None = None,  # prior turns for within-session memory
) -> dict:
    conversation = list(history) if history else []
    conversation.append({"role": "user", "content": question})
    last_claude_output = ""

    for step in range(max_steps):
        # stream response, call on_token(text) for each token
        # track last_claude_output for fallback

        if "SEARCH:" in claude_output:
            # run search(top_k=8), append results to conversation
        elif "ANSWER:" in claude_output:
            # extract text after ANSWER:, return structured dict
        else:
            # no protocol token — treat as final answer immediately
            return {"answer": claude_output.strip(), ...}

    # max steps reached — return last actual Claude output, not a hardcoded error
    return {"answer": last_claude_output.strip(), ...}
```

**System prompt rules:**
1. Never state a number not found in search results
2. Cite which document each figure came from
3. After 2 failed searches, say "not found in available documents"
4. Never use outside knowledge about Vancity
5. Max 3 searches before answering

**Output format every ANSWER must follow:**
```
**Summary** — one paragraph direct answer
**Key Figures** — markdown table (if applicable)
**Source** — which documents used
**Confidence: High / Medium / Low** — one sentence reason
```

**Verify:** Agent returns `{answer, steps, total_steps}`. Conversational questions 
("Hello") return immediately without hitting max steps.

---

### T6: CLI Entry Point
**What:** `main.py` that loads/builds index and runs question loop in terminal.

**Files:** `main.py`

**Key logic:**
```python
if index_exists():
    index, chunks = load_vector_store()
else:
    # extract all 10 PDFs → combine with section labels → chunk → build

while True:
    question = input("Your question: ").strip()
    result = agent_answer(question, index, chunks)
    # streaming output goes to terminal via print() in agent
    # print reasoning trace (steps) after answer
```

**Verify:** `python main.py` starts, loads index, answers a financial question 
with a cited figure.

---

### T7: Streamlit Chat Frontend
**What:** Browser-based chat UI with live status updates, streaming answer, 
dollar sign escaping, and within-session conversation memory.

**Files:** `app.py`

**Key behaviours:**
- `st.chat_input` + `st.chat_message` for proper chat layout
- On question submit:
  1. Status placeholder shows `*Thinking…*`
  2. As agent searches: status updates to `🔍 Searching documents for: <query>…`
  3. When ANSWER: detected: status updates to `✍️ Writing answer…`
  4. Answer streams word-by-word into a separate placeholder with `$` escaped
  5. Status clears; final clean answer replaces the stream
- Sidebar with example questions (injected via `pending_question` session state)
- Clear chat button resets both UI and memory
- `render(text)` helper escapes `$` before `$digit` / `$(` to prevent Streamlit 
  LaTeX rendering of currency values

**Dollar sign fix:**
```python
import re

def render(text: str):
    st.markdown(re.sub(r'\$(?=[\d(])', r'\\$', text))
```

**Conversation memory:**
```python
result = agent_answer(
    question,
    st.session_state.index,
    st.session_state.chunks,
    on_token=on_token,
    history=st.session_state.messages[:-1],  # all prior turns
)
```

**Run:** `streamlit run app.py`

**Verify:** 
- Ask "Compare total assets 2023 vs 2024" → get cited figures with correct formatting
- Follow up with "Why did they change?" → agent uses prior answer as context
- Dollar amounts display correctly (not as garbled math)
- Status line updates during search, disappears when answer appears

---

## Validation — Full End-to-End Check

Run these questions and verify the agent answers correctly with cited sources:

```
1. What were Vancity's total assets in 2024?
   Expected: $28.4B, source: 2024 Annual Report / Accountability Statements, High confidence

2. How did member loans change between 2023 and 2024?
   Expected: comparison table, both years cited, High confidence

3. What is Vancity's net-zero commitment?
   Expected: 2040 target, source: Climate Report, High confidence

4. What AI initiatives did Vancity pursue in 2024?
   Expected: answer from Annual Report, Medium or High confidence

5. What was Vancity's profit on Mars in 2024?
   Expected: "not found in available documents", Low confidence
   (tests hallucination prevention)

6. [Follow-up test] Ask Q1, then ask "Why did it change from 2023?"
   Expected: agent uses prior answer context, no re-introduction needed
   (tests conversation memory)
```

---

## Prompt Engineering Notes

**Why the SEARCH/ANSWER protocol:**
Simple text parsing — readable, debuggable, no JSON schema needed for a PoC.
In production, use Claude's native tool-calling API for structured reliability.

**Why the confidence signal:**
For financial data, users need to know when the agent is certain vs guessing.
Maps directly to Vancity's responsible AI principles.

**Why cite the source document:**
Auditability — stakeholders need to verify numbers come from official documents,
not hallucinated. Also surfaces when a figure was only in 2023 vs 2024 docs.

**Why cap searches at 3:**
Prevents runaway agent loops. Each search = API call = latency + cost. 
In enterprise settings, bounded behaviour is non-negotiable.

**Why Sonnet over Haiku:**
Haiku could not reliably extract figures from pipe-delimited financial table 
chunks — it would search, receive valid results, then report being unable to 
parse the numbers. Sonnet handles structured table text correctly. The cost 
difference is acceptable for a PoC; in production, caching (prompt caching) 
would reduce per-query cost significantly.