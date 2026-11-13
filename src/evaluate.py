"""Leakage-safe, same-set evaluation. Refuses unverified seed labels."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from .core import TOPICS, SENTIMENTS, clean_reviews, load_csv
from .models import ClassicClassifier, EmbeddingClassifier, scores

REQUIRED = {"id", "text", "topic", "sentiment", "human_verified"}

def read_verified(path) -> pd.DataFrame:
    evaluation = load_csv(path)
    evaluation.columns = [str(c).strip().lower() for c in evaluation.columns]
    absent = REQUIRED - set(evaluation.columns)
    if absent:
        raise ValueError(f"Evaluation CSV missing columns: {', '.join(sorted(absent))}")
    flag = evaluation["human_verified"].astype(str).str.casefold().str.strip()
    verified = evaluation.loc[flag.isin({"yes", "true", "1"})].copy()
    if len(verified) < 100:
        raise ValueError(f"Only {len(verified)} hand-reviewed items are marked verified. At least 100 are required.")
    if verified["text"].isna().any() or verified["text"].astype(str).str.strip().eq("").any():
        raise ValueError("Verified evaluation contains empty text.")
    if verified["text"].str.casefold().duplicated().any():
        raise ValueError("Verified evaluation has duplicate review texts.")
    if not verified["topic"].isin(TOPICS).all() or not verified["sentiment"].isin(SENTIMENTS).all():
        raise ValueError("Verified labels contain unknown topic or sentiment values.")
    if verified["id"].duplicated().any():
        raise ValueError("Evaluation IDs must be unique.")
    return verified

def evaluate_models(train_path, eval_path, output_dir, *, embeddings: bool = True) -> dict:
    train, _ = clean_reviews(load_csv(train_path))
    verified = read_verified(eval_path)
    if "topic" not in train.columns or "sentiment" not in train.columns:
        raise ValueError("Training CSV must contain topic and sentiment labels.")
    if not train["topic"].isin(TOPICS).all() or not train["sentiment"].isin(SENTIMENTS).all():
        raise ValueError("Training data contains unknown labels.")
    # Use normalized text to prevent cosmetic text changes leaking eval rows into training.
    eval_keys = set(verified["text"].str.strip().str.casefold())
    train = train.loc[~train["text"].str.casefold().isin(eval_keys)].copy()
    if len(train) < 50:
        raise ValueError("Too few training rows remain after evaluation leakage removal.")
    names = [("TF-IDF + Logistic Regression", ClassicClassifier)]
    if embeddings:
        names.append(("MiniLM embeddings + Logistic Regression", EmbeddingClassifier))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_results = {"verified_rows": len(verified), "training_rows": len(train), "same_eval_ids": verified["id"].astype(str).tolist(), "scores": {}}
    error_candidates = []
    for task, labels in [("topic", TOPICS), ("sentiment", SENTIMENTS)]:
        for name, constructor in names:
            model = constructor().fit(train["text"], train[task])
            predicted = model.predict(verified["text"])
            key = f"{task} | {name}"
            all_results["scores"][key] = scores(verified[task], predicted, labels)
            preds = pd.DataFrame({"id": verified["id"], "text": verified["text"], "actual": verified[task], "predicted": predicted, "task": task, "model": name})
            safe_name = f"{task}_{'tfidf' if name.startswith('TF-IDF') else 'minilm'}"
            preds.to_csv(Path(output_dir) / f"predictions_{safe_name}.csv", index=False)
            errors = preds.loc[preds["actual"] != preds["predicted"]].copy()
            for row in errors.itertuples(index=False):
                error_candidates.append({"id": str(row.id), "task": task, "model": name, "text": row.text, "actual": row.actual, "predicted": row.predicted, "why_the_model_failed": "REVIEW AND EXPLAIN THIS ERROR"})
    (Path(output_dir) / "evaluation_results.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    pd.DataFrame(error_candidates, columns=["id", "task", "model", "text", "actual", "predicted", "why_the_model_failed"]).to_csv(Path(output_dir)/"mistakes_to_analyse.csv", index=False)
    return all_results

def main():
    parser = argparse.ArgumentParser(description="Compare topic and sentiment models on >=100 human-verified held-out reviews")
    parser.add_argument("--train", default="data/reviews.csv")
    parser.add_argument("--eval", default="data/evaluation_to_verify.csv")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--classic-only", action="store_true", help="Debug classic baseline; not the full required comparison")
    args = parser.parse_args()
    try:
        report = evaluate_models(args.train, args.eval, args.output, embeddings=not args.classic_only)
    except (ValueError, RuntimeError) as exc:
        parser.exit(2, f"Evaluation not complete: {exc}\n")
    for name, result in report["scores"].items():
        print(f"{name:<48} Accuracy {result['accuracy']:.3f}, macro F1 {result['macro_f1']:.3f}")
    print(f"Saved evaluation files to {args.output}. Add explanations to at least 5 real errors.")
if __name__ == "__main__":
    main()
