import os
import re
import fitz
import faiss
import json
from config import CHUNK_WORDS, OVERLAP_WORDS, EMBED_DIM, DATA_DIR, OUTPUT_DIR, INDEX_PATH, METADATA_PATH, get_embed_model
from typing import List, Dict

EMBED_MODEL = get_embed_model()

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
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"❌ Failed to open {pdf_path}: {e}")
        return []

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

def split_text_into_chunks(text, chunk_size=CHUNK_WORDS, overlap=OVERLAP_WORDS) -> List[str]:
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

def build_faiss_index(chunks: List[Dict]) -> tuple[faiss.IndexFlatIP, List[Dict]]:
    # Batch encode embeddings for efficiency
    texts = [chunk["text"] for chunk in chunks]
    embeddings = EMBED_MODEL.encode(
        texts, 
        batch_size=32, 
        convert_to_numpy=True, 
        normalize_embeddings=True
    ).astype("float32")

    # FAISS index with Inner Product (for cosine similarity w/ normalized vectors)
    # cos(θ) = A • B / (||A|| * ||B||)
    # cos(θ) = A • B when ||A|| = ||B|| = 1
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)

    metadata = [
        {
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "page": chunk["page"],
            "text": chunk["text"]
        }
        for chunk in chunks
    ]

    return index, metadata

def build_pipeline():
    """Runs ingestion, chunking, and embedding, and indexing."""
    print("📥 Loading and parsing PDFs...")
    docs = load_documents_from_folder(DATA_DIR)

    print("✂️ Chunking text...")
    chunks = chunk_documents(docs)

    print("📊 Embedding and indexing...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    index, metadata = build_faiss_index(chunks)

    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Index built with {len(metadata)} chunks.\n")
