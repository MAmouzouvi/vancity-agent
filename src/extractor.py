# This is to convert the raw PDF into text, and then extract the relevant information from the text.

# WHAT IT DOES: Opens a PDF and pulls out all readable text
# TRADEOFFS:
# - pdfplumber is great for text-heavy PDFs like reports
# - Scanned/image PDFs need OCR (optical character recognition) instead
# - Vancity's reports are text-based so pdfplumber works fine here

import pdfplumber

def extract_text(pdf_path: str) -> str:
    """
    Opens a PDF file and extracts all text, page by page.
    Returns one combined string of all the text.
    """
    all_text = ""

    with pdfplumber.open(pdf_path) as pdf:
        print(f"  → {len(pdf.pages)} pages found in {pdf_path}")

        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()

            if page_text:                        # skip image-only pages
                all_text += page_text + "\n"

    print(f"  → Extracted {len(all_text):,} characters")
    return all_text