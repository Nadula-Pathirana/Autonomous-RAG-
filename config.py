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
MAX_DISTANCE = 0.750 

# --- Paths ---
UPLOAD_DIR = "data/uploads"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "knowledge_base"              

# --- Agent ---
MAX_WEB_RESULTS = 3          # how many web results Tavily returns
AGENT_RECURSION_LIMIT = 10   # safety cap on agent loop steps


GROQ_FALLBACK_MODEL = "allam-2-7b"      # used if the primary writes a bad tool call


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

# --- Routing (V6) ---
ROUTER_HISTORY_MESSAGES = 4   # how much conversation the router sees
PROBE_DISTANCE = 0.60         # "documents clearly match" threshold; tune with scripts/calibrate_retrieval.py
MAX_EVIDENCE_CHARS = 6000     # cap on evidence sent to the LLM per source type

# --- Grounding (V6) ---
ENABLE_VERIFICATION = True    # extra LLM check that the answer is supported by the evidence

# --- Retrieval / web (smaller = fewer tokens) ---
MAX_WEB_RESULTS = 3
MAX_WEB_CHARS = 600           # NEW: max characters kept from each web result
MAX_EVIDENCE_CHARS = 4000     # was 6000: cap per source type sent to the LLM

# --- Memory (history is sent with every answer, so keep it small) ---
MAX_HISTORY_MESSAGES = 6      # was 10
MAX_HISTORY_CHARS = 3000      # was 6000
MAX_MESSAGE_CHARS = 800       # was 1500

# --- Speed and cost (V7) ---
ROUTER_MODEL = ""             # "" = use GROQ_MODEL. A small, fast model from your list is better here.
ROUTER_MAX_TOKENS = 200       # the router only writes a tiny JSON object
ROUTER_HISTORY_MESSAGES = 3
ROUTER_REASONING_EFFORT = None   # e.g. "none"; ONLY set after testing (Section 11)
ANSWER_REASONING_EFFORT = None   # e.g. "low";  ONLY set after testing (Section 11)

# --- Grounding (V7) ---
VERIFY_MODE = "off"           # "off" = no extra LLM check, "after" = check AFTER the answer is shown


def check_environment() -> tuple[list[str], list[str]]:
    """Return (errors, warnings) about missing configuration."""
    errors, warnings = [], []
    if not GROQ_API_KEY:
        errors.append("GROQ_API_KEY is missing. The assistant can't answer questions without it.")
    if not GOOGLE_API_KEY:
        errors.append("GOOGLE_API_KEY is missing. Documents can't be embedded or searched without it.")
    if not TAVILY_API_KEY:
        warnings.append("TAVILY_API_KEY is missing. Web search will be unavailable.")
    if not GROQ_FALLBACK_MODEL:
        warnings.append("No fallback model configured (GROQ_FALLBACK_MODEL). Model failures will show errors.")
    return errors, warnings