import os
import json
import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Load env + OpenAI
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
CHAT_MODEL = "gpt-3.5-turbo"

# Load embedding model and FAISS index
TOP_K = 5
EMBED_DIM = 384
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
INDEX_PATH = "../outputs/faiss_index.idx"
METADATA_PATH = "../outputs/metadata.json"

_index = None
_metadata = None

def load_index_and_metadata():
    global _index, _metadata

    if _index is None or _metadata is None:
        _index = faiss.read_index(INDEX_PATH)

        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            _metadata = json.load(f)

    return _index, _metadata

# ----------------------------
# Retrieval Functions
# ----------------------------

def embed_query(query: str) -> np.ndarray:
    """Embed the user question into a vector."""
    emb = EMBED_MODEL.encode(query)
    return (emb / np.linalg.norm(emb)).astype(np.float32)

def retrieve_chunks(query: str, k: int = TOP_K) -> list:
    """Embed the query and return top-k matching chunks with metadata."""
    query_vector = embed_query(query).reshape(1, -1)
    scores, indices = _index.search(query_vector, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if score >= 0.4 and idx < len(_metadata):
            results.append(_metadata[idx])

    return results

# ----------------------------
# Augmentation + Generation Functions
# ----------------------------

def build_prompt(question: str, context_chunks: list) -> str:
    """Format the prompt using retrieved context and user question."""
    context = "\n\n---\n\n".join(
        f"[{c['source']} (Page {c['page']})]\n{c['text']}" for c in context_chunks
    )

    prompt = f"""
        You are a helpful assistant answering questions using context from Hunger Games-themed 
        source documents.

        Use ONLY the provided context below to answer the user's question. If the answer is not 
        in the context, respond exactly with: "The chunks were not in your favor."

        Do not use your own knowledge or make assumptions.

        Context:
        {context}

        Question:
        {question}

        Answer:
    """.strip()

    return prompt

def generate_answer(question: str, top_k: int = TOP_K) -> dict:
    """Retrieve relevant chunks and generate an answer using OpenAI API."""
    context_chunks = retrieve_chunks(question, k=top_k)
    prompt = build_prompt(question, context_chunks)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert on Hunger Games literature."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    answer = response.choices[0].message.content.strip()

    return {
        "question": question,
        "answer": answer,
        "sources": [
            {"source": c["source"], "page": c["page"]}
            for c in context_chunks
        ],
        "chunk_ids": [c["chunk_id"] for c in context_chunks]
    }
