"""TF-IDF + one-vs-rest logistic regression statute-ID baseline.

The point of this model is not accuracy -- it is to establish the floor that the
InLegalBERT head has to beat, on CPU, in a couple of minutes. It also gives the
LangGraph slice something to run before any GPU training has happened.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fir.config import ARTIFACT_DIR, pipeline
from fir.data.ilsi import FactInstance
from fir.statute.labels import LabelSpace

DEFAULT_PATH = ARTIFACT_DIR / "models" / "tfidf_statute.joblib"


@dataclass(slots=True)
class TfidfStatuteClassifier:
    """Sparse linear multi-label classifier over ILSI fact text."""

    label_space: LabelSpace
    vectorizer: object = None
    clf: object = None
    threshold: float = 0.5
    train_seconds: float = 0.0
    vectorizer_mode: str = "tfidf"

    name: str = "tfidf+ovr-logistic"

    # -- training -----------------------------------------------------------
    def fit(self, train: list[FactInstance]) -> "TfidfStatuteClassifier":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier

        cfg = pipeline()["statute_id"]["tfidf"]
        t0 = time.perf_counter()

        # Two vectorizers, chosen by config or by corpus size:
        #
        #   tfidf    TfidfVectorizer -- builds a vocabulary dict, prunes to
        #            max_features. Exact, but the dict of every 1-2-gram in the
        #            corpus exists in RAM before pruning. On 42,834 x ~7,100-char
        #            documents that dict alone killed the process on this 7.7 GB
        #            box, twice.
        #   hashing  HashingVectorizer -> TfidfTransformer. No vocabulary, constant
        #            memory regardless of corpus size; the cost is hash collisions
        #            at 2^18 features (negligible here) and no min_df pruning.
        #
        # `vectorizer: auto` uses tfidf below `hashing_above_docs` documents and
        # hashing at or above it, so the 8k slice stays exactly reproducible and
        # the full corpus stays inside RAM.
        mode = str(cfg.get("vectorizer", "auto"))
        if mode == "auto":
            mode = "hashing" if len(train) >= int(cfg.get("hashing_above_docs", 20000)) else "tfidf"
        self.vectorizer_mode = mode

        if mode == "hashing":
            from sklearn.feature_extraction.text import (
                HashingVectorizer,
                TfidfTransformer,
            )
            from sklearn.pipeline import make_pipeline

            self.vectorizer = make_pipeline(
                HashingVectorizer(
                    n_features=int(cfg.get("hashing_features", 2**18)),
                    ngram_range=tuple(cfg["ngram_range"]),
                    alternate_sign=False,
                    norm=None,
                    strip_accents="unicode",
                    lowercase=True,
                    dtype=np.float32,
                ),
                TfidfTransformer(sublinear_tf=cfg["sublinear_tf"]),
            )
        else:
            self.vectorizer = TfidfVectorizer(
                ngram_range=tuple(cfg["ngram_range"]),
                min_df=cfg["min_df"],
                max_features=cfg["max_features"],
                sublinear_tf=cfg["sublinear_tf"],
                strip_accents="unicode",
                lowercase=True,
                dtype=np.float32,   # halves the parent's matrix; liblinear upcasts per worker anyway
            )
        print(f"    vectorizer: {mode}", flush=True)
        if mode == "hashing":
            # Chunked: HashingVectorizer is stateless, so transform in slices and
            # vstack -- peak RSS is one slice plus the accumulated CSR, not the
            # whole corpus's intermediate lists at once. Then fit the IDF pass.
            import scipy.sparse as sp

            hv, idf = self.vectorizer.steps[0][1], self.vectorizer.steps[1][1]
            chunk = int(cfg.get("hashing_chunk_docs", 8000))
            parts = []
            for start in range(0, len(train), chunk):
                parts.append(hv.transform([i.text for i in train[start : start + chunk]]))
                print(f"      vectorized {min(start + chunk, len(train)):6d}/{len(train)}", flush=True)
            X = sp.vstack(parts, format="csr")
            del parts
            X = idf.fit_transform(X)
        else:
            X = self.vectorizer.fit_transform([i.text for i in train])
        y = self.label_space.encode(train)

        # Parallel across 100 independent binary problems is the whole reason
        # this finishes on CPU in the time budget -- but capped (config n_jobs):
        # each liblinear worker may hold its own copy of X, and this box has
        # 24 cores against ~7 GB of RAM.
        self.clf = OneVsRestClassifier(
            LogisticRegression(
                C=cfg["C"],
                max_iter=cfg["max_iter"],
                solver="liblinear",
                class_weight="balanced",
            ),
            n_jobs=int(cfg.get("n_jobs", 4)),
        )
        self.clf.fit(X, y)
        self.train_seconds = time.perf_counter() - t0
        return self

    # -- inference ----------------------------------------------------------
    def predict_proba(self, texts: list[str], chunk: int = 4000) -> np.ndarray:
        """Chunked: the WSL VM on this box reboots when a process's Python heap
        grows past ~2 GB (PROGRESS.md 6e); vectorising 13k long documents in one
        call gets there. Each chunk's sparse matrix is scored and freed."""
        if self.vectorizer is None or self.clf is None:
            raise RuntimeError("classifier is not fitted")
        out = []
        for i in range(0, len(texts), chunk):
            X = self.vectorizer.transform(texts[i : i + chunk])
            out.append(np.asarray(self.clf.predict_proba(X), dtype=np.float32))
        return np.concatenate(out, axis=0) if len(out) > 1 else out[0]

    def predict_sections(
        self, text: str, threshold: float | None = None
    ) -> list[str]:
        """Predict IPC section numbers for one document."""
        probs = self.predict_proba([text])[0]
        return self.label_space.decode(probs, threshold or self.threshold)

    def top_k(self, text: str, k: int = 5) -> list[tuple[str, float]]:
        return self.label_space.top_k(self.predict_proba([text])[0], k=k)

    # -- persistence --------------------------------------------------------
    def save(self, path: str | Path = DEFAULT_PATH) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "labels": self.label_space.labels,
                "vectorizer": self.vectorizer,
                "clf": self.clf,
                "threshold": self.threshold,
                "train_seconds": self.train_seconds,
                "vectorizer_mode": self.vectorizer_mode,
            },
            path,
            compress=3,
        )
        return path

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PATH) -> "TfidfStatuteClassifier":
        import joblib

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"no trained TF-IDF model at {path}. "
                "Run: python -m harness.run_statute_baseline"
            )
        blob = joblib.load(path)
        return cls(
            label_space=LabelSpace(labels=blob["labels"]),
            vectorizer=blob["vectorizer"],
            clf=blob["clf"],
            threshold=blob.get("threshold", 0.5),
            train_seconds=blob.get("train_seconds", 0.0),
            vectorizer_mode=blob.get("vectorizer_mode", "tfidf"),
        )
