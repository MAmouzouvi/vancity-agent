# Vancity AI Research Assistant — Product Spec

## Why

The problem it solves: Vancity publishes thousands of pages of annual reports, 
financial statements, climate reports, and accountability documents. Extracting 
specific insights requires hours of manual reading. This agent answers questions 
instantly, cites its sources, and flags when it isn't confident — which is 
exactly the kind of responsible AI behaviour Vancity values.

---

## What

A document Q&A agent that:
- Ingests Vancity's 2023 and 2024 Annual Reports, Consolidated Financial 
  Statements, Climate Reports, Accountability Statements, and Sustainability 
  Issuance Reports (10 PDFs total)
- Answers natural language questions by retrieving relevant document chunks 
  and synthesizing a response using Claude
- Runs an agentic loop — Claude decides what to search, how many times, 
  and when it has enough information to answer
- Returns structured output with a summary, key figures table, source citation, 
  and a confidence signal
- Streams the answer word-by-word once Claude begins writing, so the response 
  feels instant; shows live status during search steps
- Maintains conversation memory within a session so follow-up questions have context
- Persists the vector index to disk so startup after the first run is ~1 second

---

## Constraints

### Must
- Use Claude (Anthropic API) as the LLM
- Use local sentence-transformers embeddings (free, private, no API cost)
- Use FAISS for vector storage and similarity search
- Save index to disk after first build — never re-embed unless documents change
- Every answer must include a confidence signal (High / Medium / Low)
- Every answer must cite which document the information came from
- Agent must cap search iterations at a fixed limit to prevent runaway loops
- Streaming via on_token callback — answer streams word-by-word with dollar signs 
  escaped to prevent LaTeX rendering in Streamlit

### Must Not
- Never state a number or fact not found in the retrieved document chunks
- Never use outside knowledge about Vancity — only what is in the documents
- Never store the API key in code — use .env only
- Do not rebuild the vector index on every run

### Out of Scope (v1)
- User authentication
- Multi-user support
- Persistent chat history across sessions (memory resets on page refresh)
- Fine-tuning any model
- Real-time document ingestion
- Deployment to cloud (Azure, AWS, etc.)

---

## Architecture

```
10 Vancity PDFs
      ↓
  extract_text()        pdfplumber — tables extracted as pipe-delimited rows,
                        remaining page text extracted separately to avoid duplication
      ↓
  split_chunks()        800 char chunks, 150 char overlap (~2200 chunks total)
      ↓
  embed()               sentence-transformers all-MiniLM-L6-v2 (384 dims)
      ↓
  build_vector_store()  FAISS IndexFlatL2 — saved to data/faiss.index
      ↓
  agent_answer()        Claude agent loop — SEARCH / ANSWER protocol
                        on_token callback streams tokens to UI
                        history parameter carries conversation memory
      ↓
  app.py                Streamlit chat interface — live status + streaming answer
```

---

## Tech Stack

| Component        | Tool                              | Why                                                        |
|------------------|-----------------------------------|------------------------------------------------------------|
| LLM              | claude-sonnet-4-6 (Anthropic)     | Strong financial table parsing; haiku struggled with numbers|
| Embeddings       | sentence-transformers MiniLM-L6   | Free, local, no API cost, 384 dims                         |
| Vector store     | FAISS (faiss-cpu)                 | Fast exact search, runs locally                            |
| PDF extraction   | pdfplumber                        | Table-aware extraction preserves row/column context        |
| Frontend         | Streamlit                         | Fast to build, clean chat UI, pure Python                  |
| Config           | python-dotenv                     | Keeps API key out of code                                  |

---

## Documents Indexed

| Year | Document                          |
|------|-----------------------------------|
| 2024 | Annual Report                     |
| 2024 | Consolidated Financial Statements |
| 2024 | Climate Report                    |
| 2024 | Accountability Statements         |
| 2024 | Sustainability Issuance Report    |
| 2023 | Annual Report                     |
| 2023 | Consolidated Financial Statements |
| 2023 | Climate Report                    |
| 2023 | Accountability Statements         |
| 2023 | Sustainability Issuance Report    |

---

## Tradeoffs & Design Decisions

**Local embeddings vs API embeddings**
Chose sentence-transformers over OpenAI/Azure embeddings for the PoC. Free, 
fast, private, and runs offline. In production at Vancity, would switch to 
Azure OpenAI embeddings to stay within their existing Azure stack.

**FAISS vs a managed vector DB**
FAISS runs in memory — perfect for ~2200 chunks. For production with more 
documents or multi-user access, would use Azure AI Search or Pinecone.

**Text-based SEARCH/ANSWER protocol vs native tool calling**
The SEARCH/ANSWER protocol is readable and debuggable for a PoC. In production, 
would use Claude's native tool-calling API (structured JSON) which is more 
reliable and easier to log and audit.

**Sonnet vs Haiku**
Started with Haiku for speed and cost. Switched to Sonnet after finding Haiku 
could not reliably extract figures from financial table chunks — it would 
acknowledge the search returned results but report being unable to parse the 
numbers. Sonnet handles table-structured text correctly.

**Table-aware PDF extraction**
The original extractor used `page.extract_text()` only. Financial statement 
tables came out as unstructured text with numbers detached from their labels 
(e.g. `28.4 28.8 28.3` with no year headers). Updated to use `extract_tables()` 
first, converting each row to pipe-delimited format 
(`Total assets | $28.4B | $28.8B | $28.3B`), then extracting non-table text 
separately to avoid duplication. This gives Claude the row/column context it 
needs to cite figures accurately.

**top_k 8 vs 5**
Increased from 5 to 8 chunks per search. Broad comparison questions (e.g. 
"compare all key metrics 2023 vs 2024") need more chunks to cover multiple 
table rows that were split across chunk boundaries.

**Streaming with dollar sign escaping**
Streamlit's markdown renderer treats `$` as a LaTeX math delimiter. Streaming 
partial text like `$515.8 million` caused characters to render individually in 
math mode. Fix: escape `$` signs followed by digits or `(` using regex on every 
streaming frame. Status updates (search query, writing phase) are shown in a 
separate placeholder so they never interfere with the answer text.

**Chunk size 800 / overlap 150**
800 chars captures full paragraphs including context. 150 char overlap 
(~1-2 sentences) prevents answers being split across chunk boundaries.

---

## Responsible AI Considerations

- **Confidence signal**: Every answer includes High/Medium/Low confidence 
  and a reason — prevents users from trusting uncertain outputs
- **Source citation**: Every answer cites which document it came from — 
  provides auditability and traceability
- **Hallucination prevention**: Strict system prompt rules prevent Claude 
  from using outside knowledge — answers are grounded in retrieved text only
- **Scope limiting**: Agent caps search iterations to prevent runaway loops 
  and unpredictable behaviour
- **Graceful fallback**: If Claude responds without SEARCH: or ANSWER: 
  protocol tokens, the agent treats the response as the final answer rather 
  than looping to max steps

---

## What v2 Would Look Like

- Native tool-calling API instead of text-based SEARCH/ANSWER protocol
- Azure OpenAI embeddings instead of local sentence-transformers
- Azure AI Search instead of in-memory FAISS
- Add document metadata (year, document type) to each chunk for filtered retrieval
- Evaluation framework — measure retrieval precision and answer accuracy
- Persistent cross-session memory
- Deploy on Azure Container Apps with proper auth