# main.py
#
# Entry point. Loads all 4 PDFs, builds the agent, runs the question loop.

from src.extractor import extract_text
from src.chunker import split_chunks
from src.vector_store import build_vector_store, load_vector_store, index_exists
from src.agent import agent_answer

def main():

    if index_exists():
        print("\n=== Loading saved index (instant) ===\n")
        index, chunks = load_vector_store()

    else:
        print("\n=== First run: building index (one time only) ===\n")

        # ── 2024 Documents ──────────────────────────────────
        print("Loading 2024 documents...")
        text_2024_annual          = extract_text("data/vancity-2024-annual-report.pdf")
        text_2024_financial       = extract_text("data/2024-consolidated-financial-statements.pdf")
        text_2024_climate         = extract_text("data/Vancity-2024-climate-report.pdf")
        text_2024_accountability  = extract_text("data/Vancity-2024-accountability-statements.pdf")
        text_2024_sustainability  = extract_text("data/Vancity-2024-sustainability-issuance-report.pdf")

        # ── 2023 Documents ──────────────────────────────────
        print("Loading 2023 documents...")
        text_2023_annual          = extract_text("data/2023-annual-report.pdf")
        text_2023_financial       = extract_text("data/2023-consolidated-financial-statements.pdf")
        text_2023_climate         = extract_text("data/2023-climate-report.pdf")
        text_2023_accountability  = extract_text("data/2023-accountability-statements.pdf")
        text_2023_sustainability  = extract_text("data/2023-sustainability-issuance-report.pdf")

        combined_text = f"""
=== 2024 ANNUAL REPORT ===
{text_2024_annual}

=== 2024 CONSOLIDATED FINANCIAL STATEMENTS ===
{text_2024_financial}

=== 2024 CLIMATE REPORT ===
{text_2024_climate}

=== 2024 ACCOUNTABILITY STATEMENTS ===
{text_2024_accountability}

=== 2024 SUSTAINABILITY ISSUANCE REPORT ===
{text_2024_sustainability}

=== 2023 ANNUAL REPORT ===
{text_2023_annual}

=== 2023 CONSOLIDATED FINANCIAL STATEMENTS ===
{text_2023_financial}

=== 2023 CLIMATE REPORT ===
{text_2023_climate}

=== 2023 ACCOUNTABILITY STATEMENTS ===
{text_2023_accountability}

=== 2023 SUSTAINABILITY ISSUANCE REPORT ===
{text_2023_sustainability}
"""
        chunks = split_chunks(combined_text)
        index, chunks = build_vector_store(chunks)

    # ── Question Loop ────────────────────────────────────
    print("\n=== Vancity Research Agent Ready ===")
    print("Covers: 2023 + 2024 | Annual, Financial, Climate,")
    print("        Accountability + Sustainability Reports")
    print("Type 'quit' to exit\n")

    while True:
        question = input("Your question: ").strip()

        if not question:
            continue
        if question.lower() == "quit":
            break

        print("\nThinking...\n")
        print("-" * 55)
        result = agent_answer(question, index, chunks)

        # Answer already streamed above — just show the trace
        print("-" * 55)
        print("\nReasoning trace:")
        for step in result["steps"]:
            print(f"  → {step}")
        print(f"(Completed in {result['total_steps']} steps)\n")

if __name__ == "__main__":
    main()