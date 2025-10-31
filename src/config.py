import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from functools import lru_cache

load_dotenv()

# ----------------------------
# File Paths
# ----------------------------
DATA_DIR = "../data"
OUTPUT_DIR = "../outputs"
INDEX_PATH = os.path.join(OUTPUT_DIR, "faiss_index.idx")
METADATA_PATH = os.path.join(OUTPUT_DIR, "metadata.json")

# ----------------------------
# Embedding / Chunking Params
# ----------------------------
CHUNK_SIZE = 250
OVERLAP = 50
EMBED_DIM = 384
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# ----------------------------
# Retrieval / Generation Params
# ----------------------------
TOP_K = 5
CHAT_MODEL_NAME = "gpt-3.5-turbo"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ----------------------------
# Cached Models / Clients
# ----------------------------

@lru_cache(maxsize=1)
def get_embed_model():
    return SentenceTransformer(EMBED_MODEL_NAME)

@lru_cache(maxsize=1)
def get_openai_client():
    return OpenAI(api_key=OPENAI_API_KEY)
