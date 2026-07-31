import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def evaluate(model_name: str, test_cases: list, k_values: list = (1, 5, 10)) -> dict:
    from rag_core import retrieve, config
    
    # Override model
    original_model = config.EMBED_MODEL
    config.EMBED_MODEL = model_name
    
    # Force reload
    retrieve._model = None
    
    metrics = {f"recall@{k}": [] for k in k_values}
    metrics["mrr"] = []
    metrics["ndcg@10"] = []
    latencies = []
    
    for case in test_cases:
        question = case["question"]
        target_id = case.get("grounded_college_id")
        if not target_id: continue
        
        t0 = time.perf_counter()
        try:
            hits = retrieve.search(question, top_k=max(k_values))
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        latencies.append(time.perf_counter() - t0)
        
        ids = [h.get("college_id") for h in hits]
        
        # Recall@k
        for k in k_values:
            metrics[f"recall@{k}"].append(1.0 if target_id in ids[:k] else 0.0)
        
        # MRR
        if target_id in ids:
            rank = ids.index(target_id) + 1
            metrics["mrr"].append(1.0 / rank)
        else:
            metrics["mrr"].append(0.0)
        
        # nDCG@10 (binary relevance)
        relevance = [1.0 if cid == target_id else 0.0 for cid in ids[:10]]
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))
        idcg = 1.0  # only 1 relevant doc
        metrics["ndcg@10"].append(dcg / idcg if idcg > 0 else 0.0)
    
    # Restore
    config.EMBED_MODEL = original_model
    retrieve._model = None
    
    # Aggregate
    summary = {}
    for key, values in metrics.items():
        summary[key] = round(float(np.mean(values)), 4) if values else 0.0
    summary["latency_p50_ms"] = round(float(np.median(latencies)) * 1000, 1) if latencies else 0
    summary["latency_p95_ms"] = round(float(np.percentile(latencies, 95)) * 1000, 1) if latencies else 0
    summary["model"] = model_name
    summary["cases"] = len(test_cases)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="evals/embedding_test_set.json")
    ap.add_argument("--models", nargs="+", default=[
        "intfloat/multilingual-e5-small",
        "BAAI/bge-m3",
        "intfloat/multilingual-e5-base"
    ])
    args = ap.parse_args()
    
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    
    results = []
    for model in args.models:
        print(f"\\nEvaluating {model}...")
        result = evaluate(model, cases)
        print(json.dumps(result, indent=2))
        results.append(result)
    
    # Compare
    print("\\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"{'Model':<60} {'Recall@5':>10} {'MRR':>8} {'nDCG@10':>8} {'P95ms':>8}")
    for r in results:
        print(f"{r['model'][:58]:<60} {r['recall@5']:>10.3f} {r['mrr']:>8.3f} "
              f"{r['ndcg@10']:>8.3f} {r['latency_p95_ms']:>8.1f}")


if __name__ == "__main__":
    main()
