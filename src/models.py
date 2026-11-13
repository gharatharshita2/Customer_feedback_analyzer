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

def possible_new_issues(df: pd.DataFrame, threshold: float = 0.55, max_groups: int = 3) -> list[dict]:
    """Cluster 'Other' and low-confidence cases; don't present these as facts."""
    if df.empty or "confidence" not in df or "topic" not in df:
        return []
    uncertain = df.loc[(df["topic"] == "Other") | (df["confidence"] < threshold)].head(300)
    if len(uncertain) < 3:
        return []
    try:
        vectors = TfidfVectorizer(stop_words="english", min_df=1).fit_transform(uncertain["text"].tolist())
    except ValueError:
        return []
    k = min(max_groups, max(1, len(uncertain)//3))
    groups = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(vectors)
    result = []
    for group_id in sorted(set(groups)):
        members = uncertain.iloc[np.flatnonzero(groups == group_id)]
        if len(members) < 2:
            continue
        result.append({"group": int(group_id)+1, "count": len(members), "examples": members[["id", "raw_text"]].head(4).to_dict("records")})
    return sorted(result, key=lambda r: -r["count"])
