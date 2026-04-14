# Claude agent loop

# Claude reads the question
#→ decides: "I need to search for X"
#→ searches, gets chunks back
#→ reads chunks, decides: "I need to search for Y too"
#→ searches again
#→ reads everything, decides: "I have enough to answer"
#→ gives final answer


# agent.py
#
# WHAT IT DOES: Claude agent that searches documents and answers questions
# TRADEOFFS:
# - max_steps=6: prevents infinite loops, enough for complex questions
# - top_k=4: 4 chunks per search = ~3200 chars of context, good balance
# - Confidence signal: critical for enterprise use — flags uncertain answers
# - Simple SEARCH/ANSWER protocol: easy to parse, reliable, debuggable

import os
import time
import anthropic
from dotenv import load_dotenv
from src.vector_store import search

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are a financial research agent for Vancity Credit Union.
You have access to Vancity's 2023 and 2024 Annual Reports, Consolidated
Financial Statements, Climate Reports, Accountability Statements, and
Sustainability Issuance Reports via a document search tool.

## YOUR TOOLS

To search the documents:
SEARCH: <your query>

To deliver your final answer:
ANSWER: <your answer>

## SEARCH STRATEGY

Single fact:
- One targeted search is enough
- Example: SEARCH: total assets 2024

Comparison across years:
- Search broadly first to catch both years at once
- Example: SEARCH: total assets revenue net income 2023 2024
- Then do a second search only if something is missing

Trend or strategic question:
- Search for the topic across both reports
- Example: SEARCH: climate commitments net zero progress 2023 2024

## STRICT RULES

1. Never state a number or fact you did not find in the search results
2. If a figure appears in results, cite which document it came from
3. If you cannot find something after 2 searches, explicitly say:
   "This information was not found in the available documents"
4. Never guess, estimate, or use outside knowledge about Vancity
5. Do not search more than 3 times — answer with what you have by then

## OUTPUT FORMAT

Structure every ANSWER like this:

**Summary**
One paragraph with the direct answer in plain language.

**Key Figures** (if applicable)
| Metric | 2023 | 2024 | Change |
|--------|------|------|--------|

**Source**
Which documents the answer is based on.

**Confidence: High / Medium / Low**
One sentence explaining why."""


def agent_answer(question: str, index, chunks: list[str]) -> dict:
    """
    Runs the agent loop with streaming so answers feel instant.
    """
    conversation = []
    steps_log = []
    max_steps = 10

    conversation.append({
        "role": "user",
        "content": question
    })

    for step in range(max_steps):

        # ── Streaming API call with retry logic ──────────────────────────────
        claude_output = ""
        max_retries = 3
        retry_delay = 1  # start with 1 second

        for attempt in range(max_retries):
            try:
                with client.messages.stream(
                        model="claude-haiku-4-5",
                        max_tokens=2048,
                        system=SYSTEM_PROMPT,
                        messages=conversation
                ) as stream:
                    for text in stream.text_stream:
                        print(text, end="", flush=True)  # prints word by word
                        claude_output += text

                break  # success, exit retry loop

            except anthropic.APIStatusError as e:
                if e.status_code == 529 or "overloaded" in str(e).lower():
                    if attempt < max_retries - 1:
                        print(f"\n⚠️  API overloaded, retrying in {retry_delay}s...", flush=True)
                        time.sleep(retry_delay)
                        retry_delay *= 2  # exponential backoff
                    else:
                        print(f"\n❌ API still overloaded after {max_retries} attempts. Please try again later.")
                        return {
                            "answer": "The API is currently overloaded. Please try again in a moment.",
                            "steps": steps_log,
                            "total_steps": step
                        }
                else:
                    raise  # re-raise other errors

        print()  # newline after streaming ends

        conversation.append({
            "role": "assistant",
            "content": claude_output
        })

        # ── Did Claude search? ──────────────────────────────
        if "SEARCH:" in claude_output:
            query = claude_output.split("SEARCH:")[1].strip().split("\n")[0]
            steps_log.append(f"Step {step + 1}: Searched → '{query}'")
            print(f"\n🔍 Searching: '{query}'...")

            results = search(query, index, chunks, top_k=5)
            context = "\n\n---\n\n".join(results)

            conversation.append({
                "role": "user",
                "content": f"Search results:\n\n{context}"
            })

        # ── Did Claude answer? ──────────────────────────────
        elif "ANSWER:" in claude_output:
            final = claude_output.split("ANSWER:")[1].strip()
            steps_log.append(f"Step {step + 1}: Generated answer")

            return {
                "answer": final,
                "steps": steps_log,
                "total_steps": step + 1
            }

    return {
        "answer": "Agent reached max steps. Try asking about a more specific figure.",
        "steps": steps_log,
        "total_steps": max_steps
    }