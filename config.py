import os
from dotenv import load_dotenv

load_dotenv()  # reads the .env file into environment variables

# --- API keys (never hardcode these) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Models ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "qwen/qwen3.8-27b"   # check Groq's docs for currently available models

# --- Chunking ---
CHUNK_SIZE = 1000      # max characters per chunk
CHUNK_OVERLAP = 150    # characters shared between neighbouring chunks

# --- Retrieval ---
TOP_K = 4              # how many chunks to retrieve per question

# --- Paths ---
UPLOAD_DIR = "data/uploads"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "knowledge_base"