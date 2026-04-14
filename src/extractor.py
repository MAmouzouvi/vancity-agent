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
    Tables are converted to a readable pipe-delimited format so
    numbers stay associated with their row/column headers.
    Returns one combined string of all the text.
    """
    all_text = ""

    with pdfplumber.open(pdf_path) as pdf:
        print(f"  → {len(pdf.pages)} pages found in {pdf_path}")

        for page in pdf.pages:
            # Extract structured tables first so numbers keep their context
            tables = page.extract_tables()
            table_bboxes = []

            for table in tables:
                if not table:
                    continue
                rows = []
                for row in table:
                    cleaned = [cell.replace("\n", " ").strip() if cell else "" for cell in row]
                    rows.append(" | ".join(cleaned))
                all_text += "\n".join(rows) + "\n\n"

                # Track table bounding boxes to avoid double-extracting their text
                for t_obj in page.find_tables():
                    table_bboxes.append(t_obj.bbox)

            # Extract non-table text (exclude areas covered by tables)
            if table_bboxes:
                remaining = page
                for bbox in table_bboxes:
                    remaining = remaining.outside_bbox(bbox)
                page_text = remaining.extract_text()
            else:
                page_text = page.extract_text()

            if page_text:
                all_text += page_text + "\n"

    print(f"  → Extracted {len(all_text):,} characters")
    return all_text