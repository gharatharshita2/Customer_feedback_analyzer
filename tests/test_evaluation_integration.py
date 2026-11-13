import json
import pandas as pd
from src import evaluate as evaluation_module
from src.models import ClassicClassifier


def test_full_same_set_comparison_with_injected_encoder(tmp_path, monkeypatch):
    """No model download needed; checks evaluation pipeline, not MiniLM quality."""
    from pathlib import Path
    project = Path(__file__).resolve().parents[1]
    draft = pd.read_csv(project / "data" / "evaluation_to_verify.csv")
    draft["human_verified"] = "yes"  # ONLY IN TEMPORARY TEST FIXTURE, never saved as real annotations.
    tmp_eval = tmp_path / "temporary_test_fixture.csv"
    draft.to_csv(tmp_eval, index=False)
    monkeypatch.setattr(evaluation_module, "EmbeddingClassifier", ClassicClassifier)
    result = evaluation_module.evaluate_models(project / "data" / "reviews.csv", tmp_eval, tmp_path / "results", embeddings=True)
    assert result["verified_rows"] == 120
    assert result["training_rows"] == 720
    assert len(result["scores"]) == 4
    assert len(set(result["same_eval_ids"])) == 120
    assert len([p for p in (tmp_path / "results").glob("predictions_*.csv")]) == 4
    data = json.loads((tmp_path / "results" / "evaluation_results.json").read_text())
    assert len(data["scores"]) == 4
