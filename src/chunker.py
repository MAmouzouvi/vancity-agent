#This is convert the extracted text into chunks, which can then be used for embedding and vectorization.


# WHAT IT DOES: Splits a long string into overlapping chunks
# TRADEOFFS:
# - chunk_size=800 characters (about 150 words): good for financial docs — captures full paragraphs
# - overlap=150: ~1-2 sentences of overlap, prevents boundary losses
# - Smaller chunks = more precise retrieval but lose context
# - Larger chunks = more context but retrieval is less targeted

def split_chunks(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """
    Splits text into overlapping chunks.

    chunk_size: how many characters per chunk
    overlap: how many characters consecutive chunks share
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        # moving forward by (chunk_size - overlap) creates the overlap
        start += chunk_size - overlap

    print(f"  → Created {len(chunks)} chunks")
    return chunks