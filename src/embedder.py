# This is to convert the chunks of text into embeddings, which can then be stored in a vector database for retrieval.

# embedder.py
#
# WHAT IT DOES: Converts text into a list of numbers (a vector)
# TRADEOFFS:
# - sentence-transformers: free, local, private. Perfect for a PoC
# - In production: I would use Azure OpenAI embeddings
# - all-MiniLM-L6-v2: small fast model, downloads once (~80MB), runs offline

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed(text: str) -> list[float]:
    """
    Converts a string into a vector of numbers representing its meaning.
    Returns a list of 384 floats (output size of the model).
    """
    vector = model.encode(text)
    return vector.tolist()