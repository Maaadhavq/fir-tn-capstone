"""Train and evaluate the statute-ID models on ILSI.

    python -m harness.run_statute_baseline                # TF-IDF + InLegalBERT, subsampled
    python -m harness.run_statute_baseline --full         # whole corpus
    python -m harness.run_statute_baseline --skip-bert    # TF-IDF only (CPU, ~minutes)

Reports micro/macro-F1 on the test split for each model. The decision threshold
is tuned on dev, never fixed at 0.5 -- see harness.metrics.tune_threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from fir.config import ARTIFACT_DIR, pipeline  # noqa: E402
from fir.data.ilsi import load_ilsi  # noqa: E402
from fir.statute.labels import LabelSpace  # noqa: E402
from harness.metrics import (  # noqa: E402
    MultiLabelScores,
    f1_table,
    score_multilabel,
    tune_threshold,
)


def _load_splits(args) -> tuple[list, list, list]:
    cfg = pipeline()["slice"]
    n_train = None if args.full else (args.max_train or cfg["max_train_docs"])
    n_dev = None if args.full else (args.max_dev or cfg["max_dev_docs"])
    n_test = None if args.full else (args.max_test or cfg["max_test_docs"])

    print("==> loading ILSI")
    t0 = time.perf_counter()
    train = load_ilsi("train", limit=n_train)
    dev = load_ilsi("dev", limit=n_dev)
    test = load_ilsi("test", limit=n_test)
    print(
        f"    train={len(train)}  dev={len(dev)}  test={len(test)}  "
        f"({time.perf_counter() - t0:.1f}s)"
    )
    avg_labels = np.mean([len(i.ipc_labels) for i in train]) if train else 0
    avg_chars = np.mean([len(i.text) for i in train]) if train else 0
    print(f"    avg labels/doc={avg_labels:.2f}   avg chars/doc={avg_chars:.0f}")
    return train, dev, test


def run_tfidf(train, dev, test, space: LabelSpace) -> tuple[str, MultiLabelScores]:
    from fir.statute.tfidf_baseline import TfidfStatuteClassifier

    print("\n==> TF-IDF + one-vs-rest logistic")
    clf = TfidfStatuteClassifier(label_space=space).fit(train)
    print(f"    fitted in {clf.train_seconds:.1f}s")

    dev_probs = clf.predict_proba([d.text for d in dev])
    thr, dev_sc = tune_threshold(space.encode(dev), dev_probs)
    clf.threshold = thr
    print(f"    dev  micro-F1={dev_sc.micro_f1:.4f}  macro-F1={dev_sc.macro_f1:.4f}  @thr={thr:.2f}")

    test_probs = clf.predict_proba([t.text for t in test])
    test_sc = score_multilabel(space.encode(test), test_probs, thr)
    print(f"    test micro-F1={test_sc.micro_f1:.4f}  macro-F1={test_sc.macro_f1:.4f}")

    path = clf.save()
    print(f"    saved -> {path.relative_to(REPO_ROOT)}")
    return clf.name, test_sc


def run_inlegalbert(train, dev, test, space: LabelSpace, args) -> tuple[str, MultiLabelScores]:
    from fir.statute.inlegalbert import (
        DEFAULT_DIR,
        InLegalBertStatuteClassifier,
        pick_device,
    )

    device = pick_device("auto")
    tag = args.tag or ""
    print(f"\n==> InLegalBERT multi-label head (device={device}{', tag=' + tag if tag else ''})")
    if device == "cpu":
        print("    WARNING: no CUDA -- this will be very slow on CPU.")

    # CLI overrides for ablations. pipeline() is lru_cached, so mutating the
    # returned dict is seen by fit(); the tag goes into the model name so the
    # report rows never collide.
    cfg = pipeline()["statute_id"]["inlegalbert"]
    if args.truncation:
        cfg["truncation"] = args.truncation
    if args.head_lr_mult is not None:
        cfg["head_lr_multiplier"] = args.head_lr_mult
    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    clf = InLegalBertStatuteClassifier(label_space=space)
    if tag:
        clf.name = f"InLegalBERT[{tag}]"
    clf.fit(train, dev=dev)
    print(f"    trained in {clf.train_seconds / 60:.1f} min")

    test_probs = clf.predict_proba([t.text for t in test])
    test_sc = score_multilabel(space.encode(test), test_probs, clf.threshold)
    print(f"    test micro-F1={test_sc.micro_f1:.4f}  macro-F1={test_sc.macro_f1:.4f}")

    path = clf.save(DEFAULT_DIR.with_name(DEFAULT_DIR.name + (f"_{tag}" if tag else "")))
    print(f"    saved -> {path.relative_to(REPO_ROOT)}")
    return clf.name, test_sc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true", help="use the entire corpus")
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-dev", type=int, default=None)
    ap.add_argument("--max-test", type=int, default=None)
    ap.add_argument("--skip-bert", action="store_true", help="TF-IDF only")
    ap.add_argument("--skip-tfidf", action="store_true")
    # ablation overrides (InLegalBERT only); --tag keeps artifacts and report rows apart
    ap.add_argument("--tag", default="", help="suffix for the BERT artifact dir and report row")
    ap.add_argument("--truncation", choices=["head", "head_tail"], default=None)
    ap.add_argument("--head-lr-mult", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    space = LabelSpace.from_vocab()
    print(f"==> label space: {len(space)} IPC sections")

    train, dev, test = _load_splits(args)
    rows: list[tuple[str, MultiLabelScores]] = []

    if not args.skip_tfidf:
        rows.append(run_tfidf(train, dev, test, space))
    if not args.skip_bert:
        rows.append(run_inlegalbert(train, dev, test, space, args))

    print("\n" + "=" * 78)
    print("STATUTE IDENTIFICATION -- test split")
    print("=" * 78)
    print(f1_table(rows))
    print(f"\n(n_test={len(test)}, n_labels={len(space)})")

    # Full-corpus and subsampled runs answer different questions and must not
    # overwrite each other's rows.
    out = ARTIFACT_DIR / "reports" / (
        "statute_baseline_full.json" if args.full else "statute_baseline.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    # Merge into any existing report rather than overwriting it. Without this,
    # `--skip-tfidf` would silently drop the TF-IDF row and the F1 table would
    # lose the very baseline the encoder is meant to be compared against.
    previous: dict = {}
    if out.exists():
        try:
            previous = json.loads(out.read_text(encoding="utf-8")).get("results", {})
        except (OSError, json.JSONDecodeError):
            previous = {}

    out.write_text(
        json.dumps(
            {
                "n_train": len(train),
                "n_dev": len(dev),
                "n_test": len(test),
                "n_labels": len(space),
                "full_corpus": bool(args.full),
                "results": previous | {
                    name: {
                        # per-row provenance: ablation rows on other subsets must not
                        # inherit the header's counts
                        "n_train": len(train),
                        "n_test": len(test),
                        "micro_f1": sc.micro_f1,
                        "macro_f1": sc.macro_f1,
                        "micro_precision": sc.micro_precision,
                        "micro_recall": sc.micro_recall,
                        "sample_f1": sc.sample_f1,
                        "threshold": sc.threshold,
                        "labels_predicted": sc.labels_predicted,
                    }
                    for name, sc in rows
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"report -> {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
