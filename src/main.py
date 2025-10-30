import os
import faiss
import json
from preprocessing import load_documents_from_folder, chunk_documents, build_faiss_index
from rag import generate_answer
from config import DATA_DIR, OUTPUT_DIR, INDEX_PATH, METADATA_PATH

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

def main():
    print("THE HUNGER GAMES Q&A")
    print("May the chunks be ever in your favor.\n")

    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        print("🔧 No existing index found. Rebuilding from data/ ...")
        build_pipeline()
    else:
        print("✅ Index and metadata found. Skipping rebuild.")

    while True:
        question = input("\nAsk your Hunger Games question (or 'q' to quit): ")
        if question.lower() == 'q':
            break

        result = generate_answer(question)

        print("\n📘 Answer:\n", result["answer"])
        print("\n📎 Sources:")
        for s in result["sources"]:
            print(f"• {s['source']} (Page {s['page']})")


if __name__ == "__main__":
    main()
