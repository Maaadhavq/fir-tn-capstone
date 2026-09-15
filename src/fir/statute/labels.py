"""The statute label space, shared by every statute-ID model.

Fixing the label order in one place matters: the TF-IDF baseline, the
InLegalBERT head, and the golden test all index into the same 100-dim vector, so
a saved model stays readable after a retrain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fir.data.ilsi import FactInstance, load_label_vocab, parse_ipc_section


@dataclass(slots=True)
class LabelSpace:
    """Bidirectional map between ILSI label strings and vector positions."""

    labels: list[str]

    @classmethod
    def from_vocab(cls) -> "LabelSpace":
        return cls(labels=load_label_vocab())

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def index(self) -> dict[str, int]:
        return {lbl: i for i, lbl in enumerate(self.labels)}

    @property
    def sections(self) -> list[str]:
        """Bare IPC section numbers in label order, e.g. ['2', '3', ..., '498A']."""
        return [parse_ipc_section(lbl) or lbl for lbl in self.labels]

    def encode(self, instances: list[FactInstance]) -> np.ndarray:
        """Gold labels -> a (n_examples, n_labels) binary matrix."""
        idx = self.index
        y = np.zeros((len(instances), len(self.labels)), dtype=np.float32)
        for i, inst in enumerate(instances):
            for lbl in inst.ipc_labels:
                j = idx.get(lbl)
                if j is not None:
                    y[i, j] = 1.0
        return y

    def decode(self, scores: np.ndarray, threshold: float) -> list[str]:
        """One score row -> the IPC section numbers above `threshold`.

        Returned in descending score order so the officer-facing list leads with
        the model's strongest call.
        """
        secs = self.sections
        hits = np.flatnonzero(np.asarray(scores) >= threshold)
        hits = hits[np.argsort(-np.asarray(scores)[hits])]
        return [secs[j] for j in hits]

    def decode_labels(self, scores: np.ndarray, threshold: float) -> list[str]:
        """Same as `decode`, but returns the full ILSI label strings."""
        hits = np.flatnonzero(np.asarray(scores) >= threshold)
        hits = hits[np.argsort(-np.asarray(scores)[hits])]
        return [self.labels[j] for j in hits]

    def top_k(self, scores: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        """The k highest-scoring sections with their scores.

        A fallback for the officer UI when nothing clears the threshold -- an
        empty section list is less useful than a ranked shortlist.
        """
        scores = np.asarray(scores)
        secs = self.sections
        order = np.argsort(-scores)[:k]
        return [(secs[j], float(scores[j])) for j in order]
