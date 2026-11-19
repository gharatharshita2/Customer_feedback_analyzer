"""Two comparable supervised classifiers and exploratory issue clustering."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.cluster import KMeans

EMBEDDING_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class ClassicClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("classifier", LogisticRegression(max_iter=1400, class_weight="balanced", random_state=42)),
        ])
    def fit(self, texts, labels):
        self.pipeline.fit(list(texts), list(labels))
        return self
    def predict(self, texts):
        return self.pipeline.predict(list(texts))
    def confidence(self, texts):
        return self.pipeline.predict_proba(list(texts)).max(axis=1)

class EmbeddingClassifier:
    def __init__(self, encoder=None):
        if encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("Install optional embedding dependencies with: python -m pip install -r requirements-embeddings.txt") from exc
            encoder = SentenceTransformer(EMBEDDING_NAME)
        self.encoder = encoder
        self.classifier = LogisticRegression(max_iter=1400, class_weight="balanced", random_state=42)
    def _vectors(self, texts):
        return self.encoder.encode(list(texts), show_progress_bar=False, normalize_embeddings=True)
    def fit(self, texts, labels):
        self.classifier.fit(self._vectors(texts), list(labels))
        return self
    def predict(self, texts):
        return self.classifier.predict(self._vectors(texts))
    def confidence(self, texts):
        return self.classifier.predict_proba(self._vectors(texts)).max(axis=1)

def scores(y_true, y_pred, labels):
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": list(labels),
    }

def possible_new_issues(
    df: pd.DataFrame,
    threshold: float = 0.55,
    max_groups: int = 3,
    known_variations: tuple[str, ...] = (),
    min_reports: int = 3,
    known_issues: tuple[str, ...] = (),
    similarity_threshold: float = 0.20,
) -> list[dict]:
    """Group uncertain negative reports, then label business-confirmed known patterns.

    This is candidate discovery, not proof of novelty. Rules come from the user,
    and examples remain traceable to original feedback records.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    required = {"id", "text", "raw_text", "topic", "confidence", "sentiment"}
    if df.empty or not required.issubset(df.columns):
        return []
    if not 0 <= threshold <= 1 or min_reports < 2 or max_groups < 1:
        raise ValueError("Invalid clustering settings")

    candidates = df.loc[
        df["sentiment"].eq("negative")
        & (df["topic"].eq("Other") | df["confidence"].lt(threshold))
    ].copy()
    approved_variations = [" ".join(rule.casefold().split()) for rule in known_variations if str(rule).strip()]
    if approved_variations:
        normalized = candidates["raw_text"].fillna("").astype(str).map(lambda t: " ".join(t.casefold().split()))
        mask = normalized.map(lambda t: not any(rule in t for rule in approved_variations))
        candidates = candidates.loc[mask].copy()
    candidates = candidates.drop_duplicates(subset="id").head(300)
    if len(candidates) < min_reports:
        return []

    try:
        vectors = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1).fit_transform(
            candidates["text"].fillna("").astype(str)
        )
    except ValueError:
        return []
    k = min(max_groups, max(1, len(candidates) // min_reports))
    # Identical TF-IDF feature rows are not distinct cluster centres.
    # This avoids K-Means warnings on highly templated synthetic examples.
    unique_vectors = len({
        (tuple(vectors.getrow(i).indices), tuple(vectors.getrow(i).data.round(6)))
        for i in range(vectors.shape[0])
    })
    k = min(k, unique_vectors)
    group_labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(vectors)
    known_phrases = [" ".join(rule.casefold().split()) for rule in known_issues if str(rule).strip()]
    results = []

    for label in sorted(set(group_labels)):
        positions = np.flatnonzero(group_labels == label)
        n = len(positions)
        if n < min_reports:
            continue
        sims = cosine_similarity(vectors[positions])
        mean_similarity = float((sims.sum() - n) / (n * (n - 1)))
        if mean_similarity < similarity_threshold:
            continue
        members = candidates.iloc[positions]
        text = [" ".join(str(v).casefold().split()) for v in members["raw_text"].fillna("")]
        counts = [(phrase, sum(phrase in t for t in text)) for phrase in known_phrases]
        matched, count = max(counts, key=lambda x: x[1], default=("", 0))
        is_known = count >= min_reports and count / n >= 0.60
        results.append({
            "group": int(label) + 1,
            "count": n,
            "examples": members[["id", "raw_text"]].head(4).to_dict("records"),
            "status": "Known recurring issue" if is_known else "Possible new issue - needs review",
            "matched_known_issue": matched if is_known else None,
            "reason": f"{n} similar negative reports (similarity: {mean_similarity:.2f})",
            "similarity": round(mean_similarity, 3),
        })
    return sorted(results, key=lambda row: -row["count"])
