"""Is InLegalBERT losing to TF-IDF because it is a worse model, or because it
only sees the first 512 tokens of a 7,180-character document?

This is the question that decides the next move. If truncation is the cause, the
fix is hierarchical/chunked encoding and the encoder approach is still right. If
not, the encoder approach itself needs rethinking.

The experiment: train the *same* TF-IDF baseline twice -- once on full documents,
once on documents truncated to exactly the 512 wordpieces InLegalBERT sees
(truncated with InLegalBERT's own tokenizer, then decoded back to text, so the
budget is identical rather than approximated by a character count).

    python -m harness.ablate_truncation

If truncated TF-IDF collapses toward the InLegalBERT score, truncation is the
dominant factor and the model is not the problem.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fir.config import ARTIFACT_DIR, pipeline  # noqa: E402
from fir.data.ilsi import load_ilsi  # noqa: E402
from fir.statute.labels import LabelSpace  # noqa: E402
from fir.statute.tfidf_baseline import TfidfStatuteClassifier  # noqa: E402
from harness.metrics import score_multilabel, tune_threshold  # noqa: E402


def truncate_like_bert(texts: list[str], max_len: int = 512) -> list[str]:
    """Cut each document to the exact wordpiece budget InLegalBERT receives."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
    out = []
    for t in texts:
        ids = tok.encode(t, add_special_tokens=False, truncation=True, max_length=max_len)
        out.append(tok.decode(ids, skip_special_tokens=True))
    return out


def evaluate(train_texts, dev_texts, test_texts, train, dev, test, space, label):
    clf = TfidfStatuteClassifier(label_space=space)
    # fit() reads .text off the instances, so swap the text in place
    for inst, txt in zip(train, train_texts):
        inst.text = txt
    clf.fit(train)

    thr, _ = tune_threshold(space.encode(dev), clf.predict_proba(dev_texts))
    sc = score_multilabel(space.encode(test), clf.predict_proba(test_texts), thr)
    print(
        f"  {label:<34s} micro-F1 {sc.micro_f1:.4f}   macro-F1 {sc.macro_f1:.4f}   "
        f"labels {sc.labels_predicted:3d}"
    )
    return sc


def main() -> int:
    cfg = pipeline()["slice"]
    space = LabelSpace.from_vocab()

    print("==> loading ILSI")
    train = load_ilsi("train", limit=cfg["max_train_docs"])
    dev = load_ilsi("dev", limit=cfg["max_dev_docs"])
    test = load_ilsi("test", limit=cfg["max_test_docs"])

    full_train = [i.text for i in train]
    full_dev = [i.text for i in dev]
    full_test = [i.text for i in test]

    print("==> truncating to InLegalBERT's 512-wordpiece budget")
    cut_train = truncate_like_bert(full_train)
    cut_dev = truncate_like_bert(full_dev)
    cut_test = truncate_like_bert(full_test)

    kept = sum(len(a) for a in cut_train) / max(sum(len(a) for a in full_train), 1)
    print(f"    truncation keeps {kept * 100:.1f}% of the training text\n")

    print("TF-IDF, identical model, two input budgets:")
    full_sc = evaluate(full_train, full_dev, full_test, train, dev, test, space,
                       "full document")
    cut_sc = evaluate(cut_train, cut_dev, cut_test, train, dev, test, space,
                      "truncated to 512 wordpieces")

    drop = full_sc.micro_f1 - cut_sc.micro_f1
    print(f"\n  micro-F1 lost to truncation alone: {drop:.4f} "
          f"({drop / full_sc.micro_f1 * 100:.1f}% relative)")

    # compare against the encoder, if it has been trained
    rep = ARTIFACT_DIR / "reports" / "statute_baseline.json"
    if rep.exists():
        res = json.loads(rep.read_text(encoding="utf-8")).get("results", {})
        bert = res.get("InLegalBERT")
        if bert:
            print(f"  InLegalBERT (also 512-limited)      micro-F1 {bert['micro_f1']:.4f}"
                  f"   macro-F1 {bert['macro_f1']:.4f}")
            if cut_sc.micro_f1 <= bert["micro_f1"]:
                print("\n  => Truncation fully explains the gap: TF-IDF given the same")
                print("     512-token budget performs no better than the encoder.")
            elif drop > (full_sc.micro_f1 - bert["micro_f1"]) * 0.5:
                print("\n  => Truncation is the dominant factor, but not the whole story.")
            else:
                print("\n  => Truncation is NOT the main cause; the encoder setup itself lags.")

    out = ARTIFACT_DIR / "reports" / "truncation_ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "n_train": len(train),
                "n_test": len(test),
                "text_retained_fraction": kept,
                "full_document": {"micro_f1": full_sc.micro_f1, "macro_f1": full_sc.macro_f1},
                "truncated_512": {"micro_f1": cut_sc.micro_f1, "macro_f1": cut_sc.macro_f1},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nreport -> {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
