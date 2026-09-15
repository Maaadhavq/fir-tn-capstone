"""The vertical-slice proof: F1 table + WER number, from real data.

    python -m harness.run_slice              # both halves, subsampled
    python -m harness.run_slice --full       # whole ILSI corpus
    python -m harness.run_slice --text-only  # skip ASR
    python -m harness.run_slice --asr-only   # skip statute training

Runs the two mini-pipelines from IMPLEMENTATION.md §4 and then demonstrates the
composed LangGraph graph on worked examples, so the output shows both the
metrics and the officer-facing decisions those metrics feed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fir.config import ARTIFACT_DIR  # noqa: E402
from fir.statute.ipc_bns_map import IpcBnsMap  # noqa: E402

RULE = "=" * 78

#: Worked examples for the composed graph. Real ILSI-style facts, chosen to show
#: an auto-applied case, a held-back case, and the CSR branch.
DEMO_CASES = [
    (
        "dowry death (ILSI-style facts)",
        "The deceased was married five years ago. After the marriage she was "
        "treated with cruelty by her husband and his relatives for demand of "
        "dowry, and on 8.5.2010 they committed her murder at the matrimonial home.",
    ),
    (
        "theft from a shop",
        "The accused entered the complainant's shop at night and stole a mobile "
        "phone and cash from the counter before fleeing.",
    ),
    (
        "public servant demanding a bribe",
        "A public servant demanded gratification from the complainant to process "
        "a routine application.",
    ),
    (
        "spoken Tamil (ITN: numbers as words, year-less date)",
        "மே மாதம் எட்டாம் தேதி இரவு பத்து மணிக்கு அடையாளம் தெரியாத ஒருவர் என் கடையில் இருந்து "
        "ஐம்பதாயிரம் ரூபாய் மற்றும் என் போன் திருடிச் சென்றார். என் எண் 9876543210.",
    ),
]


def _run(module: str, extra: list[str]) -> int:
    cmd = [sys.executable, "-m", module, *extra]
    print(f"\n$ {' '.join(cmd[2:])}\n", flush=True)
    return subprocess.call(cmd, cwd=REPO_ROOT)


def demo_graph() -> None:
    """Run the composed LangGraph graph on the worked examples."""
    from fir.orchestrator.graph import build_slice_graph, format_result, run_text
    from fir.statute.tfidf_baseline import TfidfStatuteClassifier

    print("\n" + RULE)
    print("COMPOSED PIPELINE  (text -> statute-ID -> IPC/BNS -> cognizability)")
    print(RULE)

    try:
        clf = TfidfStatuteClassifier.load()
    except FileNotFoundError as exc:
        print(f"  skipped: {exc}")
        return

    graph = build_slice_graph(classifier=clf, ipc_map=IpcBnsMap.load())
    first_form = None
    for title, narrative in DEMO_CASES:
        print(f"\n[{title}]")
        state = run_text(graph, narrative)
        print(format_result(state))
        if first_form is None and state.get("bns_sections"):
            first_form = state.get("if1_text")

    if first_form:
        print("\n" + RULE)
        print("STAGE D -- the first case above, rendered onto the fixed IF-1 form")
        print(RULE)
        print(first_form)


def mapping_coverage() -> None:
    cov = IpcBnsMap.load().coverage()
    print("\n" + RULE)
    print("IPC -> BNS MAPPING COVERAGE")
    print(RULE)
    print(f"  rows in table          : {cov['total']}")
    print(f"  auto-applicable        : {cov['auto_applicable']}  (confidence=high, not flagged)")
    print(f"  held for human review  : {cov['needs_review']}")
    print(f"  non-BNS targets        : {cov['non_bns_targets']}  (OMITTED / VERIFY / PC Act)")
    print("  NOTE: source is a third-party pocket directory, not the MHA gazette.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--text-only", action="store_true", help="skip the ASR half")
    ap.add_argument("--asr-only", action="store_true", help="skip statute training")
    ap.add_argument("--skip-bert", action="store_true", help="TF-IDF only")
    ap.add_argument("--n-clips", type=int, default=None)
    ap.add_argument(
        "--report-only",
        action="store_true",
        help="re-print the summary from existing reports without retraining",
    )
    args = ap.parse_args()
    if args.report_only:
        args.asr_only = args.text_only = True  # skip both training halves

    t0 = time.perf_counter()
    print(RULE)
    print("FIR-TN VERTICAL SLICE  (IMPLEMENTATION.md §4)")
    print(RULE)

    failures: list[str] = []

    if not args.asr_only:
        extra = []
        if args.full:
            extra.append("--full")
        if args.skip_bert:
            extra.append("--skip-bert")
        if _run("harness.run_statute_baseline", extra) != 0:
            failures.append("statute baseline")

    if not args.text_only:
        extra = ["--n", str(args.n_clips)] if args.n_clips else []
        if _run("harness.run_asr_baseline", extra) != 0:
            failures.append("ASR baseline")

    mapping_coverage()
    if not args.asr_only or args.report_only:
        demo_graph()

    # --- the headline numbers, gathered in one place ---
    print("\n" + RULE)
    print("SLICE SUMMARY")
    print(RULE)
    stat_path = ARTIFACT_DIR / "reports" / "statute_baseline.json"
    asr_path = ARTIFACT_DIR / "reports" / "asr_baseline.json"

    shown = False
    for label, path in (("subsample", stat_path),
                        ("FULL CORPUS", stat_path.with_name("statute_baseline_full.json"))):
        if path.exists():
            rep = json.loads(path.read_text(encoding="utf-8"))
            print(f"  statute-ID [{label}]  (n_train={rep['n_train']}, n_test={rep['n_test']}, "
                  f"{rep['n_labels']} IPC sections)")
            for name, r in rep["results"].items():
                print(f"    {name:<26s} micro-F1 {r['micro_f1']:.4f}   macro-F1 {r['macro_f1']:.4f}")
            shown = True
    if not shown:
        print("  statute-ID  : not run")

    if asr_path.exists():
        rep = json.loads(asr_path.read_text(encoding="utf-8"))
        print(
            f"  ASR         ({rep['n_clips']} FLEURS-ta clips, "
            f"{rep['device']}/{rep['compute_type']})"
        )
        print(f"    WER {rep['wer'] * 100:.2f}%    CER {rep['cer'] * 100:.2f}%    RTF {rep['rtf']:.3f}")
    else:
        print("  ASR         : not run")

    print(f"\n  total wall clock: {(time.perf_counter() - t0) / 60:.1f} min")
    if failures:
        print(f"  FAILED: {', '.join(failures)}")
        return 1
    print("  slice complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
