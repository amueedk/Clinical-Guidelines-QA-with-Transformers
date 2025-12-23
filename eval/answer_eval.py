"""Answer evaluation: measure grounding and keyword coverage.

Usage:
    python eval/answer_eval.py --test_file eval/answer_questions.json --top_k 8
"""
import argparse
import json
from pathlib import Path
import sys
import requests

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def evaluate_answers(test_questions, api_url="http://localhost:8000/query", top_k=8):
    """Evaluate generated answers for keyword coverage and grounding."""
    results = []
    total_keywords = 0
    found_keywords = 0
    
    for item in test_questions:
        question = item["question"]
        expected_keywords = [kw.lower() for kw in item.get("expected_keywords", [])]
        
        try:
            # Query API
            response = requests.post(api_url, json={"question": question, "top_k": top_k}, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            answer = data.get("answer", "").lower()
            citations = data.get("citations", [])
            
            # Check keyword coverage
            matched = [kw for kw in expected_keywords if kw in answer]
            coverage = len(matched) / len(expected_keywords) if expected_keywords else 0.0
            
            total_keywords += len(expected_keywords)
            found_keywords += len(matched)
            
            # Check grounding (answer should not be "I don't know")
            grounded = "don't know" not in answer.lower() and len(answer) > 20
            
            results.append({
                "question": question,
                "answer": data.get("answer", ""),
                "expected_keywords": expected_keywords,
                "matched_keywords": matched,
                "coverage": coverage,
                "grounded": grounded,
                "num_citations": len(citations)
            })
            
        except Exception as e:
            print(f"Error querying '{question}': {e}")
            results.append({
                "question": question,
                "answer": "",
                "expected_keywords": expected_keywords,
                "matched_keywords": [],
                "coverage": 0.0,
                "grounded": False,
                "num_citations": 0,
                "error": str(e)
            })
    
    avg_coverage = found_keywords / total_keywords if total_keywords > 0 else 0.0
    grounded_count = sum(1 for r in results if r.get("grounded"))
    grounded_rate = grounded_count / len(results) if results else 0.0
    
    return {
        "avg_keyword_coverage": avg_coverage,
        "grounded_rate": grounded_rate,
        "total_questions": len(results),
        "grounded_count": grounded_count,
        "results": results
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, default="eval/answer_questions.json")
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--api_url", type=str, default="http://localhost:8000/query")
    parser.add_argument("--out", type=str, default="eval/answer_results.json")
    args = parser.parse_args()
    
    test_path = Path(args.test_file)
    if not test_path.exists():
        print(f"Test file not found: {test_path}")
        return
    
    test_questions = json.loads(test_path.read_text(encoding="utf-8"))
    
    print(f"Evaluating answers for {len(test_questions)} questions...")
    print(f"API: {args.api_url}, top_k: {args.top_k}\n")
    
    eval_results = evaluate_answers(test_questions, args.api_url, args.top_k)
    
    print(f"\n=== Results ===")
    print(f"Average keyword coverage: {eval_results['avg_keyword_coverage']:.2%}")
    print(f"Grounded answers: {eval_results['grounded_count']} / {eval_results['total_questions']} ({eval_results['grounded_rate']:.2%})")
    
    # Save results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(eval_results, indent=2, ensure_ascii=False), encoding="utf-8")
    
    print(f"\nDetailed results saved to: {out_path}")
    
    # Show sample answers
    print("\n=== Sample answers ===")
    for r in eval_results["results"][:3]:
        print(f"\nQ: {r['question']}")
        print(f"A: {r['answer'][:200]}...")
        print(f"Coverage: {r['coverage']:.0%}, Grounded: {r['grounded']}")


if __name__ == "__main__":
    main()
