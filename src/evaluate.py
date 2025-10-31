import os
import json
import numpy as np
from sentence_transformers import util
from config import get_embed_model
from rag import generate_answer
from config import TOP_K

EMBED_MODEL = get_embed_model()

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

    if not relevant_set:
        return None, None

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
        "precision": round(precision, 2) if precision is not None else None,
        "recall": round(recall, 2) if recall is not None else None,
        "sources": result["sources"]
    }

def run_evaluation(path="questions.json", output_file="tests/eval_results.txt"):
    os.makedirs("tests", exist_ok=True)
    
    with open(os.path.join("tests", path), "r", encoding="utf-8") as f:
        questions = json.load(f)

    scores = []
    precisions = []
    recalls = []

    with open(output_file, "w", encoding="utf-8") as out:
        out.write("🔍 Evaluating RAG System...\n\n")

        for q in questions:
            result = evaluate_question(q)
            scores.append(result)
            
            precision = result["precision"]
            recall = result["recall"]

            if precision is not None:
                precisions.append(precision)
                recalls.append(recall)

            out.write(f"🧠 Q: {result['question']}\n")
            out.write(f"✅ Expected: {result['expected_answer']}\n")
            out.write(f"🤖 Generated: {result['generated_answer']}\n")
            out.write(f"📐 Cosine Similarity: {result['similarity']}\n")
            out.write(f"📊 Precision@{TOP_K}: {precision}\n")
            out.write(f"📊 Recall@{TOP_K}: {recall}\n")
            out.write(f"📚 Sources Used: {set([s['source'] + ' (pg ' + str(s['page']) + ')' for s in result['sources']])}\n")
            out.write("-" * 60 + "\n")

        # Average metrics
        avg_similarity = np.mean([r["similarity"] for r in scores])
        avg_precision = np.mean(precisions) if precisions else 0
        avg_recall = np.mean(recalls) if recalls else 0

        out.write("\n📈 Overall Evaluation\n")
        out.write(f"Avg Cosine Similarity: {round(avg_similarity, 2)}\n")
        out.write(f"Avg Precision@{TOP_K}: {round(avg_precision, 2)}\n")
        out.write(f"Avg Recall@{TOP_K}: {round(avg_recall, 2)}\n")

    print(f"✅ Evaluation results saved to: {output_file}")

if __name__ == "__main__":
    run_evaluation()
