import os
import re
import fitz
import faiss
import json
import numpy as np
from config import CHUNK_SIZE, OVERLAP, EMBED_DIM, EMBED_MODEL_NAME
from sentence_transformers import SentenceTransformer
from typing import List, Dict

EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME)

# Patterns that usually appear at the top of academic PDFs
COVER_PATTERNS = [
    r"recommended citation",
    r"retrieved from",
    r"digital commons",
    r"this (essay|article) is brought to you",
    r"university",
    r"college",
    r"department of",
    r"issn",
    r"doi",
]
COVER_REGEX = re.compile("|".join(COVER_PATTERNS), flags=re.IGNORECASE)

REFERENCE_PATTERNS = [
    r"^works? cited\b",
    r"^references\b",
    r"^bibliography\b",
    r"^sources\b",
]
REFERENCE_REGEX = re.compile("|".join(REFERENCE_PATTERNS), flags=re.IGNORECASE | re.MULTILINE)

# ----------------------------
# Ingestion Functions
# ----------------------------

def looks_like_noncontent_page(text: str) -> bool:
    """Detect pages that are pure metadata or TOC."""
    lower = text.lower()

    # Skip common page types
    if any(kw in lower for kw in ["recommended citation", "table of contents", "copyright", "retrieved from", "by an authorized editor"]):
        return True
    
    return False

def trim_cover_from_page1(text) -> str:
    """Remove top metadata lines but keep actual essay if it starts on same page."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    cleaned = []
    found_body = False

    for ln in lines:
        # Detect when real prose begins (sentence-like)
        if not found_body:
            # Skip lines that match metadata
            if COVER_REGEX.search(ln):
                continue

            # Detect likely body: starts with capital, ends with .,?! or has 10+ words
            if re.match(r"^[A-Z].*[\.!?]$", ln) or len(ln.split()) > 10:
                found_body = True
                cleaned.append(ln)
        else:
            cleaned.append(ln)

    return "\n".join(cleaned).strip()

def remove_references_section(text: str) -> str:
    """
    Truncate text after 'Works Cited' or similar heading.
    Keeps only the essay body before references.
    """
    match = REFERENCE_REGEX.search(text)

    if match:
        cutoff = match.start()
        truncated = text[:cutoff].strip()
        return truncated, match
    
    return text, match

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []

    try:
        for page_idx in range(len(doc)):
            page_num = page_idx + 1
            text = doc[page_idx].get_text("text").strip()

            if not text:
                continue

            # Skip clear non-content pages
            if looks_like_noncontent_page(text):
                print(f"🚫 Skipping non-content page {page_num} of {os.path.basename(pdf_path)}")
                continue

            # Clean first page if needed
            if page_num == 1:
                text = trim_cover_from_page1(text)
                print(f"🧹 Cleaned cover text from page 1 of {os.path.basename(pdf_path)}")
            
            # Remove references if it's on last or second to last page
            if page_num >= len(doc) - 1:
                text, match = remove_references_section(text)
                if match:
                    print(f"✂️ Removed references section from page {page_num} of {os.path.basename(pdf_path)}")
            
            pages.append((page_num, text))
    finally:
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
