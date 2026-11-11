from functools import lru_cache

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import EMBEDDING_MODEL, GOOGLE_API_KEY

#This Python file is all about creating a function for reusing an embedding model. 
@lru_cache(maxsize=1)
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Load the embedding model once and reuse it (loading is slow)."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY
    )