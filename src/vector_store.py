# This is to store the embeddings in a vector database, and perform search

# vector_store.py
#
# WHAT IT DOES: Embeds all chunks, builds a FAISS index, enables search
# TRADEOFFS:
# - IndexFlatL2: exact search, perfect for ~1800 chunks
# - For millions of chunks, I would use approximate search (IndexIVFFlat)
# - Storing index in memory: fine for a PoC, would use a vector DB in prod (Azure AI Search)

import os
import faiss
import numpy as np
import pickle
from src.embedder import embed

INDEX_PATH  = "data/faiss.index"
CHUNKS_PATH = "data/chunks.pkl"


def index_exists() -> bool:
    """
    Checks if a saved index already exists on disk.
    If both files exist, we can skip rebuilding.
    """
    return os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH)


def build_vector_store(chunks: list[str]):
    """
    Embeds all chunks and builds a FAISS index.
    Saves to disk so you never rebuild unless documents change.
    """
    print("  → Embedding all chunks (first time only)...")

    embeddings = []
    for i, chunk in enumerate(chunks):
        vector = embed(chunk)
        embeddings.append(vector)
        if i % 100 == 0:
            print(f"     {i}/{len(chunks)} chunks embedded...")

    embeddings_array = np.array(embeddings, dtype='float32')

    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)

    # Save both to disk
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, 'wb') as f:
        pickle.dump(chunks, f)

    print(f"  → Saved to disk. {index.ntotal} chunks indexed.")
    return index, chunks


def load_vector_store():
    """
    Loads the saved index from disk instantly.
    Called on every run after the first.
    """
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, 'rb') as f:
        chunks = pickle.load(f)
    print(f"  → Loaded from disk. {index.ntotal} chunks ready.")
    return index, chunks


def search(question: str, index, chunks: list[str], top_k: int = 5) -> list[str]:
    """
    Finds the top_k chunks most semantically similar to the question.
    """
    question_vector = np.array([embed(question)], dtype='float32')
    distances, indices = index.search(question_vector, top_k)
    return [chunks[i] for i in indices[0]]