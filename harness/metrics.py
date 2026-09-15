"""Evaluation metrics, shared by every stage of the pipeline.

Multi-label statute identification reports micro- and macro-F1 because they say
different things on a long-tailed label set: micro is dominated by the frequent
sections (302, 379, 498A), macro weights the rare ones equally and is the honest
signal for whether the model handles the tail at all.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class MultiLabelScores:
    """Micro/macro/sample F1 for a multi-label prediction set."""

    micro_f1: float
    macro_f1: float
    micro_precision: float
    micro_recall: float
    macro_precision: float
    macro_recall: float
    sample_f1: float
    threshold: float
    n_examples: int
    n_labels: int
    labels_predicted: int

    def as_row(self, name: str) -> str:
        return (
            f"{name:<28s} {self.micro_f1:>9.4f} {self.macro_f1:>9.4f} "
            f"{self.micro_precision:>9.4f} {self.micro_recall:>9.4f} "
            f"{self.threshold:>7.2f}"
        )


def score_multilabel(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> MultiLabelScores:
    """Score binarised multi-label predictions at a given threshold.

    Implemented directly on numpy rather than via sklearn so the zero-division
    convention is explicit: a label with no true and no predicted positives
    scores 0, not 1. Treating a never-seen label as perfect would inflate
    macro-F1 on exactly the tail we care about measuring.
    """
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_prob) >= threshold

    tp = np.logical_and(y_true, y_pred).sum(axis=0).astype(float)
    fp = np.logical_and(~y_true, y_pred).sum(axis=0).astype(float)
    fn = np.logical_and(y_true, ~y_pred).sum(axis=0).astype(float)

    # micro: pool the counts across labels first
    mi_tp, mi_fp, mi_fn = tp.sum(), fp.sum(), fn.sum()
    mi_p = mi_tp / (mi_tp + mi_fp) if (mi_tp + mi_fp) else 0.0
    mi_r = mi_tp / (mi_tp + mi_fn) if (mi_tp + mi_fn) else 0.0
    mi_f = 2 * mi_p * mi_r / (mi_p + mi_r) if (mi_p + mi_r) else 0.0

    # macro: per-label first, then average
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
        r = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
        f = np.where((p + r) > 0, 2 * p * r / (p + r), 0.0)

    # sample-averaged: per-example F1, the closest proxy for "did this one case
    # get the right set of sections"
    s_tp = np.logical_and(y_true, y_pred).sum(axis=1).astype(float)
    s_pred = y_pred.sum(axis=1).astype(float)
    s_true = y_true.sum(axis=1).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        sp = np.where(s_pred > 0, s_tp / s_pred, 0.0)
        sr = np.where(s_true > 0, s_tp / s_true, 0.0)
        sf = np.where((sp + sr) > 0, 2 * sp * sr / (sp + sr), 0.0)

    return MultiLabelScores(
        micro_f1=float(mi_f),
        macro_f1=float(f.mean()),
        micro_precision=float(mi_p),
        micro_recall=float(mi_r),
        macro_precision=float(p.mean()),
        macro_recall=float(r.mean()),
        sample_f1=float(sf.mean()),
        threshold=float(threshold),
        n_examples=int(y_true.shape[0]),
        n_labels=int(y_true.shape[1]),
        labels_predicted=int((y_pred.sum(axis=0) > 0).sum()),
    )


def tune_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    lo: float = 0.05,
    hi: float = 0.95,
    step: float = 0.05,
) -> tuple[float, MultiLabelScores]:
    """Sweep the decision threshold for best micro-F1.

    Never leave this at 0.5. ILSI averages a handful of positive labels out of
    100, so a 0.5 cut predicts almost nothing and micro-F1 collapses.
    """
    best_t, best = lo, None
    for t in np.arange(lo, hi + 1e-9, step):
        sc = score_multilabel(y_true, y_prob, float(t))
        if best is None or sc.micro_f1 > best.micro_f1:
            best_t, best = float(t), sc
    assert best is not None
    return best_t, best


def f1_table(rows: list[tuple[str, MultiLabelScores]]) -> str:
    """Render the F1 table printed by `make slice`."""
    head = (
        f"{'model':<28s} {'micro-F1':>9s} {'macro-F1':>9s} "
        f"{'micro-P':>9s} {'micro-R':>9s} {'thr':>7s}"
    )
    sep = "-" * len(head)
    body = "\n".join(sc.as_row(name) for name, sc in rows)
    return f"{head}\n{sep}\n{body}"


# --------------------------------------------------------------------------
# ASR metrics
# --------------------------------------------------------------------------

_PUNCT_RE = re.compile(
    r"[.,!?;:\"'`‘’“”()\[\]{}<>/\\|@#$%^&*_+=~।॥-]"
)
_WS_RE = re.compile(r"\s+")


def normalise_tamil(text: str) -> str:
    """Normalise a Tamil / code-switched transcript before scoring.

    Without this, WER is dominated by artefacts rather than recognition errors:

    * NFC composition -- Tamil vowel signs have multiple valid encodings, and an
      uncomposed reference would count every such word as wrong.
    * Punctuation, including the Devanagari danda used in some Indic text.
    * Case folding, which only affects the Latin half of a code-switched line.
    """
    text = unicodedata.normalize("NFC", text)
    text = _PUNCT_RE.sub(" ", text)
    text = text.lower()
    return _WS_RE.sub(" ", text).strip()


@dataclass(slots=True)
class AsrScores:
    wer: float
    cer: float
    n_clips: int
    n_ref_words: int
    audio_seconds: float = 0.0
    rtf: float = 0.0  # real-time factor: processing time / audio duration

    def as_report(self, name: str) -> str:
        lines = [
            f"{name}",
            f"  clips        : {self.n_clips}",
            f"  ref words    : {self.n_ref_words}",
            f"  WER          : {self.wer:.4f}  ({self.wer * 100:.2f}%)",
            f"  CER          : {self.cer:.4f}  ({self.cer * 100:.2f}%)",
        ]
        if self.audio_seconds:
            lines.append(f"  audio        : {self.audio_seconds:.1f}s")
            lines.append(f"  RTF          : {self.rtf:.3f}")
        return "\n".join(lines)


def score_asr(
    references: list[str],
    hypotheses: list[str],
    audio_seconds: float = 0.0,
    elapsed_seconds: float = 0.0,
) -> AsrScores:
    """Corpus-level WER/CER via jiwer, after Tamil normalisation.

    Corpus-level, not the mean of per-clip rates: jiwer pools edit distances
    over the whole set, so one short clip with a bad transcript cannot dominate
    the number the way a per-clip average would.
    """
    import jiwer

    refs = [normalise_tamil(r) for r in references]
    hyps = [normalise_tamil(h) for h in hypotheses]

    # jiwer errors on empty references; drop those pairs and count what remains.
    pairs = [(r, h) for r, h in zip(refs, hyps) if r]
    if not pairs:
        raise ValueError("no non-empty references to score")
    refs, hyps = [p[0] for p in pairs], [p[1] for p in pairs]

    return AsrScores(
        wer=float(jiwer.wer(refs, hyps)),
        cer=float(jiwer.cer(refs, hyps)),
        n_clips=len(refs),
        n_ref_words=sum(len(r.split()) for r in refs),
        audio_seconds=audio_seconds,
        rtf=(elapsed_seconds / audio_seconds) if audio_seconds else 0.0,
    )
