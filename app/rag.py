# app/rag.py

import os
import pickle
import faiss
from faiss import IndexIDMap
from langchain_community.embeddings import OpenAIEmbeddings
from app.config import OPENAI_API_KEY, VECTOR_STORE_PATH
import numpy as np

def to_float32_vec(vec_list: list[float]) -> np.ndarray:
    """Convert a Python list of floats to a 1×D float32 NumPy vector."""
    return np.array(vec_list, dtype="float32")

# 1. Initialize embeddings client
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# 2. Load or create FAISS index
if os.path.exists(VECTOR_STORE_PATH):
    # load existing index + id map
    index = faiss.read_index(VECTOR_STORE_PATH)
else:
    # create a new flat index for 1536-dim embeddings, wrapped in an ID map
    dim = 1536
    flat_index = faiss.IndexFlatL2(dim)
    index = IndexIDMap(flat_index)

# 3. Simple helper to index one document
def index_document(text: str, doc_id: int):
    # embed_query returns a list of floats, so cast it to np.float32
    vec = to_float32_vec(embeddings.embed_query(text))
    index.add_with_ids(np.array([vec]).astype("float32"), np.array([doc_id]))
    faiss.write_index(index, VECTOR_STORE_PATH)

# 4. Simple retrieval
def retrieve(query: str, k: int = 3):
    q_vec = to_float32_vec(embeddings.embed_query(query))
    D, I = index.search(np.array([q_vec]), k)
    return I[0]  # list of doc_ids

if __name__ == "__main__":
    # 1) index a toy document
    index_document("Retrieval-Augmented Generation combines LLMs with vector search.", doc_id=1)
    # 2) retrieve against that same concept
    sample = "What is RAG?"
    print("Top docs:", retrieve(sample))
