import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from rag import generate_answer
from config import TOP_K

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# Evaluation Metrics
# ----------------------------

def cosine_similarity(a: str, b: str) -> float:
    """
    Calculate answer relevance by finding the cosine similarity between two texts.
    """
    emb1 = EMBED_MODEL.encode(a, convert_to_tensor=True)
    emb2 = EMBED_MODEL.encode(b, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb1, emb2).item()

def precision_recall(retrieved, relevant):
    """
    Compute precision and recall based on (source, page) pairs.
    """
    retrieved_set = {(s["source"], s["page"]) for s in retrieved}
    relevant_set = {(s["source"], s["page"]) for s in relevant}

    intersection = retrieved_set & relevant_set

    precision = len(intersection) / len(retrieved_set) if retrieved_set else 0
    recall = len(intersection) / len(relevant_set) if relevant_set else 0

    return precision, recall

def evaluate_question(q: dict):
    """
    Evaluate one question:
    - Generate answer
    - Compute cosine similarity
    - Compute retrieval precision & recall (page-level)
    """

    result = generate_answer(q["question"])
    cosine = cosine_similarity(result["answer"], q["expected_answer"])
    precision, recall = precision_recall(result["sources"], q.get("relevant_sources", []))

    return {
        "question": q["question"],
        "expected_answer": q["expected_answer"],
        "generated_answer": result["answer"],
        "similarity": round(cosine, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "sources": result["sources"]
    }

def run_evaluation(path="questions.json"):
    with open(os.path.join("tests", path), "r", encoding="utf-8") as f:
        questions = json.load(f)

    scores = []

    print("🔍 Evaluating RAG System...\n")

    for q in questions:
        result = evaluate_question(q)
        scores.append(result)

        print(f"🧠 Q: {result['question']}")
        print(f"✅ Expected: {result['expected_answer']}")
        print(f"🤖 Generated: {result['generated_answer']}")
        print(f"📐 Cosine Similarity: {result['similarity']}")
        print(f"📊 Precision@{TOP_K}: {result['precision']}")
        print(f"📊 Recall@{TOP_K}: {result['recall']}")
        print(f"📚 Sources Used: {[s['source'] + ' (pg ' + str(s['page']) + ')' for s in result['sources']]}")
        print("-" * 60)

    # Average metrics
    avg_similarity = np.mean([r["similarity"] for r in scores])
    avg_precision = np.mean([r["precision"] for r in scores])
    avg_recall = np.mean([r["recall"] for r in scores])

    print("\n📈 Overall Evaluation")
    print(f"Avg Cosine Similarity: {round(avg_similarity, 2)}")
    print(f"Avg Precision@{TOP_K}: {round(avg_precision, 2)}")
    print(f"Avg Recall@{TOP_K}: {round(avg_recall, 2)}")

if __name__ == "__main__":
    run_evaluation()
