import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from rag import generate_answer

model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# Evaluation Metrics
# ----------------------------

def cosine_similarity(a: str, b: str) -> float:
    """
    Calculate answer relevance by finding the cosine similarity between two texts.
    """
    emb1 = model.encode(a, convert_to_tensor=True)
    emb2 = model.encode(b, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb1, emb2).item()

def evaluate_question(q: dict, top_k: int = 5):
    result = generate_answer(q["question"], top_k=top_k)

    return {
        "question": q["question"],
        "expected_answer": q["expected_answer"],
        "generated_answer": result["answer"],
        "similarity": round(cosine_similarity(result["answer"], q["expected_answer"]), 2),
        "used_sources": result["sources"]
    }

def run_evaluation(path="questions.json", top_k=5):
    with open(os.path.join("tests", path), "r", encoding="utf-8") as f:
        questions = json.load(f)

    scores = []

    print("🔍 Evaluating RAG System...\n")

    for q in questions:
        result = evaluate_question(q, top_k=top_k)
        scores.append(result)

        print(f"🧠 Q: {result['question']}")
        print(f"✅ Expected: {result['expected_answer']}")
        print(f"🤖 Generated: {result['generated_answer']}")
        print(f"📐 Cosine Similarity: {result['similarity']}")
        print(f"📚 Chunks Used: {[s['source'] + ' (pg ' + str(s['page']) + ')' for s in result['used_sources']]}")
        print("-" * 60)

    # Average metrics
    avg_similarity = np.mean([r["similarity"] for r in scores])

    print("\n📈 Overall Evaluation")
    print(f"Avg Cosine Similarity: {round(avg_similarity, 2)}")

if __name__ == "__main__":
    run_evaluation()
