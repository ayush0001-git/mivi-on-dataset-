import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def evaluate(eval_set_path: str, output_path: str, limit: int = None):
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        context_precision, context_recall, 
        answer_relevancy, faithfulness
    )
    from rag_core import pipeline, retrieve
    
    cases = json.loads(Path(eval_set_path).read_text(encoding="utf-8"))
    cases = cases["cases"] if isinstance(cases, dict) else cases
    if limit:
        cases = cases[:limit]
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    for case in cases:
        q = case["question"]
        questions.append(q)
        gt = " ".join(
            item[0] if isinstance(item, list) else item
            for item in case.get("must_include", [])
            if isinstance(item, (str, list))
        )
        ground_truths.append(gt)
        
        try:
            result = pipeline.answer_question(q)
        except Exception as e:
            result = {"answer": f"Error: {e}", "citations": [], "answered": False}
        
        answers.append(result.get("answer", ""))
        
        try:
            from evals.run_evals import Corpus
            corpus = Corpus()
            citations = result.get("citations", [])
            if citations and corpus.backend:
                try:
                    rows = corpus.backend.q(
                        "SELECT card FROM college_agg WHERE college_id = ANY(%s)",
                        [citations]
                    )
                    contexts.append([r.get("card", "") for r in rows if r.get("card")])
                except Exception:
                    contexts.append([])
            else:
                contexts.append([])
        except ImportError:
            contexts.append([])
    
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    ds = Dataset.from_dict(data)
    
    result = ragas_evaluate(
        ds,
        metrics=[context_precision, context_recall, answer_relevancy, faithfulness]
    )
    
    print(result)
    
    Path(output_path).write_text(
        json.dumps(result.to_pandas().to_dict(orient="records"), indent=1),
        encoding="utf-8"
    )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default="evals/eval_set.json")
    ap.add_argument("--output", default="evals/ragas_report.json")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    evaluate(args.eval_set, args.output, args.limit)
