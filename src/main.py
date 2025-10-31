import os
from preprocessing import build_pipeline
from rag import generate_answer
from config import INDEX_PATH, METADATA_PATH

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
