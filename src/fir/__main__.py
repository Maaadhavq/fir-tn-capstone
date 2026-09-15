"""Command-line entry point.

    python -m fir draft "On 8.5.2026 the accused stole my phone worth Rs 15,000"
    python -m fir draft --audio clip.wav --annotate
    python -m fir draft --json "..."          # machine-readable decision + record
    python -m fir check                       # what data / models are present
    python -m fir mapping 302 498A 500        # look up IPC sections in the BNS map

Everything runs locally. `draft` never files anything; it prints a draft for the
officer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _pipeline(need_asr: bool, classifier: str = "auto"):
    from fir.orchestrator.graph import build_slice_graph
    from fir.statute import registry
    from fir.statute.ipc_bns_map import IpcBnsMap

    asr = None
    if need_asr:
        from fir.asr.whisper import WhisperAsr

        asr = WhisperAsr()
    from fir.translate import default_translator

    return build_slice_graph(
        classifier=registry.load(classifier), asr=asr, ipc_map=IpcBnsMap.load(),
        translator=default_translator(),
    )


def cmd_draft(args: argparse.Namespace) -> int:
    from fir.instantiate.render import render_if1
    from fir.orchestrator.graph import format_result, run_audio, run_text, to_decision
    from fir.schema.if1 import FirRecord

    try:
        graph = _pipeline(need_asr=bool(args.audio), classifier=args.classifier)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.audio:
        state = run_audio(graph, args.audio)
    else:
        text = args.text or sys.stdin.read()
        if not text.strip():
            print("error: no complaint text given", file=sys.stderr)
            return 2
        state = run_text(graph, text)

    if args.json:
        rec = FirRecord.model_validate(state["fir_record"])
        print(json.dumps({
            "decision": to_decision(state).model_dump(mode="json"),
            "record": rec.model_dump(mode="json"),
            "narrative_grounded": state.get("narrative_grounded"),
            "asr_flags": state.get("asr_flags", []),
            "errors": state.get("errors", []),
        }, ensure_ascii=False, indent=2))
        return 0

    print(format_result(state))
    print()
    if args.annotate:
        print(render_if1(FirRecord.model_validate(state["fir_record"]), annotate=True, lang=args.lang))
    else:
        print(state.get("if1_text") or "(no form produced)")
    return 0 if not state.get("errors") else 1


def cmd_check(args: argparse.Namespace) -> int:
    from fir.config import ARTIFACT_DIR, DATA_DIR, datasets
    from fir.data.fleurs import is_available as fleurs_ok

    ok = "ok"
    missing = "MISSING"
    rows: list[tuple[str, str, str]] = []

    ilsi = datasets()["ilsi"]
    for split in ("train", "dev", "test"):
        p = DATA_DIR / "ilsi" / ilsi["files"][split]
        rows.append((f"ILSI {split}", ok if p.exists() else missing,
                     f"{p.stat().st_size / 1e6:.0f} MB" if p.exists() else "scripts/download_ilsi.sh"))
    rows.append(("FLEURS ta_in test", ok if fleurs_ok("test") else missing,
                 "" if fleurs_ok("test") else "scripts/download_fleurs.py"))
    rows.append(("IPC->BNS map", ok if (DATA_DIR / "mapping" / "ipc_bns_map.csv").exists() else missing, ""))
    from fir.statute.cognizability import schedule

    sch = schedule()
    rows.append(("BNSS Schedule 1",
                 missing if sch.is_stub else ok,
                 f"stub ({sch.n_rows} hand-coded rows) -- drop data/statutes/bnss_schedule1.csv to replace"
                 if sch.is_stub else f"{sch.n_rows} rows from bnss_schedule1.csv"))

    from fir.statute.bns_text import bns_text

    bt = bns_text()
    rows.append(("BNS 2023 text", ok if bt else missing,
                 f"{len(bt)} sections from bns_2023.jsonl" if bt else "scripts/build_bns_text.py"))

    for name, path in (("TF-IDF statute model", ARTIFACT_DIR / "models" / "tfidf_statute.joblib"),
                       ("InLegalBERT head", ARTIFACT_DIR / "models" / "inlegalbert_statute" / "statute_head.json")):
        rows.append((name, ok if path.exists() else missing,
                     "" if path.exists() else "python -m harness.run_statute_baseline"))

    try:
        import torch

        gpu = f"{torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" \
            if torch.cuda.is_available() else "no CUDA"
    except Exception:  # noqa: BLE001
        gpu = "torch not importable"
    rows.append(("GPU", ok if "GB" in gpu else missing, gpu))
    from fir.translate.indictrans import IndicTranslator

    tr = IndicTranslator()
    rows.append(("IndicTrans2 ta->en", ok if tr.available() else missing,
                 tr.model_id if tr.available() else "scripts/download_indictrans.py (~800 MB)"))

    w = max(len(r[0]) for r in rows)
    for name, status, note in rows:
        print(f"  {name:<{w}s}  {status:<8s} {note}")
    return 0 if all(r[1] == ok for r in rows if r[0] not in ("BNSS Schedule 1", "BNS 2023 text", "IndicTrans2 ta->en")) else 1


def cmd_mapping(args: argparse.Namespace) -> int:
    from fir.statute.cognizability import classify_section
    from fir.statute.ipc_bns_map import IpcBnsMap

    m = IpcBnsMap.load()
    if not args.sections:
        cov = m.coverage()
        print(json.dumps(cov, indent=2))
        return 0
    res = m.apply(args.sections)
    for row in res.auto_applied:
        cog = classify_section(row.normalised_bns).value
        print(f"  IPC {row.ipc_section:<6s} -> {row.normalised_bns:<14s} AUTO   {cog:<15s} {row.offence}")
    for item in res.review_queue:
        print(f"  IPC {item.ipc_section:<6s} -> {str(item.suggested_bns or '-'):<14s} REVIEW {item.reason.value:<15s} {item.offence}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fir", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("draft", help="draft an IF-1 from complaint text or audio")
    d.add_argument("text", nargs="?", help="complaint text (or pipe via stdin)")
    d.add_argument("--audio", type=Path, help="audio file instead of text")
    d.add_argument("--annotate", action="store_true", help="tag machine-filled slots")
    d.add_argument("--lang", default="en", choices=["en", "ta"])
    d.add_argument("--json", action="store_true", help="emit decision + record as JSON")
    d.add_argument("--classifier", default="auto", choices=["auto", "tfidf", "bert"],
                   help="statute model (default: config default_classifier, currently tfidf)")
    d.set_defaults(fn=cmd_draft)

    c = sub.add_parser("check", help="report which data and models are present")
    c.set_defaults(fn=cmd_check)

    mp = sub.add_parser("mapping", help="look up IPC sections in the IPC->BNS map")
    mp.add_argument("sections", nargs="*", help="IPC section numbers, e.g. 302 498A")
    mp.set_defaults(fn=cmd_mapping)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
