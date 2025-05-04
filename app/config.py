# app/config.py

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env in project root

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "data/embeddings.faiss")
