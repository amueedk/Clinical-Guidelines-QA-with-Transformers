"""Retrieval evaluation: compute Recall@k on labeled test questions.

Usage:
    python eval/retrieval_eval.py --test_file eval/test_questions.json --k 5
"""
import argparse
import json
from pathlib import Path
import sys
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Add parent dir to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import api


def load_index():
    """Load FAISS index and metadata for retrieval."""
    index_path = Path("indexes/faiss.index")
    metadata_path = Path("indexes/metadata.json")
    
    if not index_path.exists():
        print(f"Index not found: {index_path}")
        print("Please run: python scripts/build_index.py")
        sys.exit(1)
    
    # Load index and metadata
    api.index = faiss.read_index(str(index_path))
    api.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    
    # Load embedding model
    print("Loading embedding model...")
    api.embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print(f"Index loaded: {len(api.metadata)} chunks")


def compute_recall_at_k(test_questions, k=5):
    """Compute Recall@k: fraction of questions where at least one relevant chunk is in top-k."""
    total = len(test_questions)
    hits = 0
    
    results = []
    for item in test_questions:
        question = item["question"]
        relevant_chunks = set(item["relevant_chunks"])
        
        # Retrieve top-k
        try:
            retrieved = api.retrieve(question, k)
            retrieved_ids = {h["id"] for h in retrieved}
            
            # Check if any relevant chunk is in top-k
            found = bool(relevant_chunks & retrieved_ids)
            if found:
                hits += 1
            
            results.append({
                "question": question,
                "relevant": list(relevant_chunks),
                "retrieved_ids": list(retrieved_ids),
                "found": found
            })
        except Exception as e:
            print(f"Error retrieving for '{question}': {e}")
            results.append({
                "question": question,
                "relevant": list(relevant_chunks),
                "retrieved_ids": [],
                "found": False,
                "error": str(e)
            })
    
    recall = hits / total if total > 0 else 0.0
    return recall, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, default="eval/test_questions.json")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", type=str, default="eval/retrieval_results.json")
    args = parser.parse_args()
    
    test_path = Path(args.test_file)
    if not test_path.exists():
        print(f"Test file not found: {test_path}")
        return
    
    test_questions = json.loads(test_path.read_text(encoding="utf-8"))
    
    # Load index and models
    load_index()
    
    print(f"\nEvaluating retrieval on {len(test_questions)} questions with k={args.k}...")
    recall, results = compute_recall_at_k(test_questions, args.k)
    
    print(f"\nRecall@{args.k}: {recall:.2%}")
    print(f"Found relevant chunks for {sum(1 for r in results if r.get('found'))} / {len(test_questions)} questions")
    
    # Save detailed results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "recall_at_k": recall,
        "k": args.k,
        "total_questions": len(test_questions),
        "hits": sum(1 for r in results if r.get("found")),
        "results": results
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    
    print(f"\nDetailed results saved to: {out_path}")


if __name__ == "__main__":
    main()
