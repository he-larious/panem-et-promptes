import json
import os
import faiss
import numpy as np
from preprocessing import build_pipeline
from config import INDEX_PATH, METADATA_PATH, TOP_K, CHAT_MODEL_NAME, get_embed_model, get_openai_client

EMBED_MODEL = get_embed_model()
client = get_openai_client()

_index = None
_metadata = None

def load_index_and_metadata():
    global _index, _metadata

    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        build_pipeline()

    try:
        if _index is None or _metadata is None:
            _index = faiss.read_index(INDEX_PATH)

            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                _metadata = json.load(f)
    except Exception as e:
        print(f"❌ Error loading FAISS index or metadata: {e}")
        print("Rebuilding from scratch...")

        build_pipeline()

        _index = faiss.read_index(INDEX_PATH)

        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            _metadata = json.load(f)

    return _index, _metadata

# ----------------------------
# Retrieval Functions
# ----------------------------

def embed_query(query: str) -> np.ndarray:
    """Embed the user question into a vector."""
    emb = EMBED_MODEL.encode([query], normalize_embeddings=True).astype(np.float32)
    return emb

def retrieve_chunks(query: str, k: int = TOP_K) -> list:
    """Embed the query and return top-k matching chunks with metadata."""
    global _index, _metadata
    if _index is None or _metadata is None:
        _index, _metadata = load_index_and_metadata()
        
    query_vector = embed_query(query)
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

    # Early exit if nothing retrieved
    if not context_chunks:
        return {
            "question": question,
            "answer": "The chunks were not in your favor.",
            "sources": []
        }
    
    prompt = build_prompt(question, context_chunks)

    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an expert on Hunger Games literature."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        answer = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Error generating answer from OpenAI: {e}")
        answer = "The chunks were not in your favor."

    # Remove duplicate (source, page) pairs
    unique_sources = {(c["source"], c["page"]) for c in context_chunks}
    sources = [{"source": s, "page": p} for s, p in unique_sources]

    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }
