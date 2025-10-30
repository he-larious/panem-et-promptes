import os
import fitz
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict

# Constants
CHUNK_SIZE = 200
OVERLAP = 50
EMBED_DIM = 384
MODEL = SentenceTransformer("all-MiniLM-L6-v2")

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



# Run this file to test the above code
if __name__ == "__main__":
    docs = load_documents_from_folder("../data")
    chunks = chunk_documents(docs)
