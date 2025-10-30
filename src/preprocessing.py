import os
import fitz
import faiss
import json
import numpy as np
from config import CHUNK_SIZE, OVERLAP, EMBED_DIM, EMBED_MODEL_NAME
from sentence_transformers import SentenceTransformer
from typing import List, Dict

EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME)

# ----------------------------
# Ingestion Functions
# ----------------------------

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []

    for page_num in range(len(doc)):
        text = doc[page_num].get_text("text")
        if text.strip():
            pages.append((page_num + 1, text.strip()))

    doc.close()
    return pages

def load_documents_from_folder(folder_path):
    docs = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            pages = extract_text_from_pdf(pdf_path)

            for page_num, text in pages:
                docs.append({
                    "source": filename,
                    "page": page_num,
                    "text": text
                })

    return docs

# ----------------------------
# Chunking Functions
# ----------------------------

def split_text_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP) -> List[str]:
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))

    return chunks

def chunk_documents(docs: List[Dict]) -> List[Dict]:
    all_chunks = []

    for doc in docs:
        chunks = split_text_into_chunks(doc["text"])

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{doc['source'].replace('.pdf', '')}_pg{doc['page']}_chunk{idx + 1}"

            all_chunks.append({
                "chunk_id": chunk_id,
                "source": doc["source"],
                "page": doc["page"],
                "text": chunk
            })

    return all_chunks

# ----------------------------
# Embedding Functions
# ----------------------------

def get_embedding(text) -> list:
    return EMBED_MODEL.encode(text).tolist()

def build_faiss_index(chunks: List[Dict]) -> tuple[faiss.IndexFlatIP, List[Dict]]:
    # FAISS index with Inner Product (for cosine similarity w/ normalized vectors)
    # cos(θ) = A • B / (||A|| * ||B||)
    # cos(θ) = A • B when ||A|| = ||B|| = 1
    index = faiss.IndexFlatIP(EMBED_DIM)
    metadata = []

    for chunk in chunks:
        emb = get_embedding(chunk["text"])
        norm_emb = np.array(emb) / np.linalg.norm(emb)
        index.add(np.array([norm_emb], dtype=np.float32))

        metadata.append({
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "page": chunk["page"],
            "text": chunk["text"]
        })

    return index, metadata

# Run this file to test the above code
if __name__ == "__main__":
    docs = load_documents_from_folder("../data")
    chunks = chunk_documents(docs)
    index, metadata = build_faiss_index(chunks)

    os.makedirs("../outputs", exist_ok=True)
    faiss.write_index(index, "../outputs/faiss_index.idx")
    with open("../outputs/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Indexed {len(metadata)} chunks.")
