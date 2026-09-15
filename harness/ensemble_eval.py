"""Does averaging TF-IDF and InLegalBERT probabilities beat either alone?

    python -m harness.ensemble_eval            # subsample test set
    python -m harness.ensemble_eval --full     # full dev/test

No training. Loads both saved classifiers, scores dev and test, and sweeps the
mixing weight α in  p = α·p_tfidf + (1-α)·p_bert  on dev (jointly with the
decision threshold), then reports test at the chosen α.

Why this is worth ten minutes: the two models err differently. TF-IDF sees the
whole document as a bag of n-grams; the encoder sees the first 512 wordpieces
with word order. Even a weak encoder can add signal a linear model lacks, and
α tells us how much. α≈1 means the encoder adds nothing yet.

Whatever artifacts are on disk are used -- if they were trained on different
subsets (8k TF-IDF vs full BERT) the result is still a valid "ensemble of the
models we have", but the report says which.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


from fir.config import ARTIFACT_DIR, pipeline  # noqa: E402
from fir.data.ilsi import load_ilsi  # noqa: E402
from fir.statute.inlegalbert import InLegalBertStatuteClassifier  # noqa: E402
from fir.statute.labels import LabelSpace  # noqa: E402
from fir.statute.tfidf_baseline import TfidfStatuteClassifier  # noqa: E402
from harness.metrics import score_multilabel, tune_threshold  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--alphas", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--bert-tag", default="", help="load artifacts/models/inlegalbert_statute_<tag>")
    args = ap.parse_args()

    cfg = pipeline()["slice"]
    n_dev = None if args.full else cfg["max_dev_docs"]
    n_test = None if args.full else cfg["max_test_docs"]

    space = LabelSpace.from_vocab()
    print("==> loading models")
    from fir.statute.inlegalbert import DEFAULT_DIR

    tfidf = TfidfStatuteClassifier.load()
    bert_dir = DEFAULT_DIR.with_name(DEFAULT_DIR.name + (f"_{args.bert_tag}" if args.bert_tag else ""))
    bert = InLegalBertStatuteClassifier.load(bert_dir)
    print(f"    tfidf: {tfidf.vectorizer_mode}, thr {tfidf.threshold:.2f}, trained {tfidf.train_seconds:.0f}s")
    print(f"    bert : {bert.truncation}, thr {bert.threshold:.2f}, "
          f"epochs {len(bert.history)}, trained {bert.train_seconds / 60:.1f} min")

    print("==> loading ILSI")
    dev = load_ilsi("dev", limit=n_dev)
    test = load_ilsi("test", limit=n_test)
    y_dev, y_test = space.encode(dev), space.encode(test)

    print("==> scoring")
    p_dev_t = tfidf.predict_proba([d.text for d in dev])
    p_dev_b = bert.predict_proba([d.text for d in dev])
    p_te_t = tfidf.predict_proba([t.text for t in test])
    p_te_b = bert.predict_proba([t.text for t in test])

    alphas = [float(a) for a in args.alphas.split(",")]
    rows = []
    print(f"\n{'alpha':>6s} {'dev micro-F1':>13s} {'dev macro-F1':>13s} {'thr':>5s}   {'test micro-F1':>14s} {'test macro-F1':>14s}")
    print("-" * 78)
    best = None
    for a in alphas:
        pd = a * p_dev_t + (1 - a) * p_dev_b
        thr, sc_dev = tune_threshold(y_dev, pd)
        pt = a * p_te_t + (1 - a) * p_te_b
        sc_te = score_multilabel(y_test, pt, thr)
        rows.append({"alpha": a, "threshold": thr,
                     "dev_micro_f1": sc_dev.micro_f1, "dev_macro_f1": sc_dev.macro_f1,
                     "test_micro_f1": sc_te.micro_f1, "test_macro_f1": sc_te.macro_f1,
                     "test_labels_predicted": sc_te.labels_predicted})
        if best is None or sc_dev.micro_f1 > best["dev_micro_f1"]:
            best = rows[-1]
        print(f"{a:>6.1f} {sc_dev.micro_f1:>13.4f} {sc_dev.macro_f1:>13.4f} {thr:>5.2f}   "
              f"{sc_te.micro_f1:>14.4f} {sc_te.macro_f1:>14.4f}")

    print("-" * 78)
    print(f"best on dev: alpha={best['alpha']:.1f}  ->  test micro-F1 {best['test_micro_f1']:.4f}, "
          f"macro-F1 {best['test_macro_f1']:.4f}")
    t_only = next(r for r in rows if r["alpha"] == 1.0)
    b_only = next(r for r in rows if r["alpha"] == 0.0)
    gain = best["test_micro_f1"] - max(t_only["test_micro_f1"], b_only["test_micro_f1"])
    print(f"gain over best single model on test: {gain:+.4f} micro-F1")
    if best["alpha"] >= 0.9:
        print("=> the encoder adds ~nothing to the linear model at present.")
    elif gain > 0.01:
        print("=> the models are complementary; the ensemble is worth serving.")

    suffix = ("_full" if args.full else "") + (f"_{args.bert_tag}" if args.bert_tag else "")
    out = ARTIFACT_DIR / "reports" / f"ensemble{suffix}.json"
    out.write_text(json.dumps({
        "n_dev": len(dev), "n_test": len(test), "bert_tag": args.bert_tag,
        "tfidf_mode": tfidf.vectorizer_mode, "bert_truncation": bert.truncation,
        "sweep": rows, "best_alpha": best["alpha"], "gain_vs_best_single": gain,
    }, indent=2), encoding="utf-8")
    print(f"report -> {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
