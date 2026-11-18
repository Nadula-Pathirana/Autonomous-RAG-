import os
from dotenv import load_dotenv

load_dotenv()  # reads the .env file into environment variables

# --- API keys (never hardcode these) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") 

# --- Models ---
EMBEDDING_MODEL = "models/gemini-embedding-001"
GROQ_MODEL = "qwen/qwen3.8-27b"   # check Groq's docs for currently available models

# --- Chunking ---
CHUNK_SIZE = 1000      # max characters per chunk
CHUNK_OVERLAP = 150    # characters shared between neighbouring chunks

# --- Retrieval ---
TOP_K = 3              # how many chunks to retrieve per question
MAX_DISTANCE = None 

# --- Paths ---
UPLOAD_DIR = "data/uploads"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "knowledge_base"              

# --- Agent ---
MAX_WEB_RESULTS = 3          # how many web results Tavily returns
AGENT_RECURSION_LIMIT = 10   # NEW: safety cap on agent loop steps


GROQ_FALLBACK_MODEL = "llama-3.3-70b-versatile"      # used if the primary writes a bad tool call


# --- Memory ---
MAX_HISTORY_MESSAGES = 10    # remember the last 10 messages (about 5 questions + 5 answers)
MAX_HISTORY_CHARS = 6000     # total character budget for the remembered history
MAX_MESSAGE_CHARS = 1500     # long old answers are shortened to this

# --- Robustness (NEW) ---
MAX_UPLOAD_MB = 20            # reject bigger PDFs
MAX_QUESTION_CHARS = 1000     # limit question length
EMBED_BATCH_SIZE = 50         # chunks sent to the embedding API at a time
RETRY_ATTEMPTS = 3            # tries for temporary failures (rate limits, timeouts)
RETRY_BASE_DELAY = 2          # seconds; doubles after each failed try
ENABLE_STREAMING = True       # show the answer as it is written

# --- Logging (NEW) ---
LOG_DIR = "logs"
LOG_LEVEL = "INFO"


def check_environment() -> tuple[list[str], list[str]]:
    """Return (errors, warnings) about missing configuration."""
    errors, warnings = [], []
    if not GROQ_API_KEY:
        errors.append("GROQ_API_KEY is missing. The assistant can't answer questions without it.")
    if not GOOGLE_API_KEY:
        errors.append("GOOGLE_API_KEY is missing. Documents can't be embedded or searched without it.")
    if not TAVILY_API_KEY:
        warnings.append("TAVILY_API_KEY is missing. Web search will be unavailable.")
    return errors, warnings