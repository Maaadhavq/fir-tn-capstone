"""Aggregate every report under artifacts/reports/ into one RESULTS.md.

    python -m harness.summarize            # writes artifacts/reports/RESULTS.md
    python -m harness.summarize --stdout

The runners each write a JSON file; this is the one place that reads them all
and renders the tables the Review-3 report needs. It never recomputes anything
-- if a number is not in a JSON report, it is not here.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fir.config import ARTIFACT_DIR  # noqa: E402

REPORTS = ARTIFACT_DIR / "reports"


def _load(name: str) -> dict | None:
    p = REPORTS / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def statute_section(out: list[str]) -> None:
    out.append("## Statute identification (ILSI, 100 IPC sections)\n")
    rows = []
    for label, fname in (("8k subsample", "statute_baseline.json"),
                         ("full corpus", "statute_baseline_full.json")):
        rep = _load(fname)
        if not rep:
            continue
        for model, r in rep["results"].items():
            rows.append((label, r.get("n_train", rep["n_train"]), r.get("n_test", rep["n_test"]), model, r))
    if not rows:
        out.append("_not run_\n")
        return
    out.append("| train set | n_train | n_test | model | micro-F1 | macro-F1 | micro-P | micro-R | labels used | thr |")
    out.append("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|")
    for label, ntr, nte, model, r in rows:
        out.append(
            f"| {label} | {ntr:,} | {nte:,} | {model} | **{r['micro_f1']:.4f}** | **{r['macro_f1']:.4f}** "
            f"| {r['micro_precision']:.4f} | {r['micro_recall']:.4f} | {r['labels_predicted']}/100 | {r['threshold']:.2f} |"
        )
    out.append("")
    out.append("Threshold tuned on dev for micro-F1. `labels used` = sections receiving at least one "
               "prediction on test; a low count means collapse onto head labels.\n")

    abl = _load("truncation_ablation.json")
    if abl:
        out.append("### Truncation ablation (TF-IDF, identical model, two input budgets)\n")
        out.append("| input | micro-F1 | macro-F1 |")
        out.append("|---|---:|---:|")
        out.append(f"| full document | {abl['full_document']['micro_f1']:.4f} | {abl['full_document']['macro_f1']:.4f} |")
        out.append(f"| truncated to 512 wordpieces | {abl['truncated_512']['micro_f1']:.4f} | {abl['truncated_512']['macro_f1']:.4f} |")
        drop = abl["full_document"]["micro_f1"] - abl["truncated_512"]["micro_f1"]
        out.append(f"\nTruncation keeps {abl['text_retained_fraction'] * 100:.1f}% of the text and costs "
                   f"{drop:.4f} micro-F1 ({drop / abl['full_document']['micro_f1'] * 100:.1f}% relative). "
                   "It does not explain the encoder gap.\n")


def ensemble_section(out: list[str]) -> None:
    for path in sorted(REPORTS.glob("ensemble*.json")):
        rep = _load(path.name)
        if not rep:
            continue
        label = "full corpus" if "_full" in path.name else "8k subsample"
        if rep.get("bert_tag"):
            label += f", BERT[{rep['bert_tag']}]"
        out.append(f"### Ensemble sweep — {label} (p = α·TF-IDF + (1−α)·BERT)\n")
        out.append(f"TF-IDF `{rep['tfidf_mode']}`, BERT `{rep['bert_truncation']}`; "
                   f"n_dev={rep['n_dev']:,}, n_test={rep['n_test']:,}. α and threshold chosen on dev.\n")
        out.append("| α | dev micro-F1 | test micro-F1 | test macro-F1 | thr |")
        out.append("|---:|---:|---:|---:|---:|")
        for r in rep["sweep"]:
            mark = " **←**" if r["alpha"] == rep["best_alpha"] else ""
            out.append(f"| {r['alpha']:.1f} | {r['dev_micro_f1']:.4f} | {r['test_micro_f1']:.4f} "
                       f"| {r['test_macro_f1']:.4f} | {r['threshold']:.2f}{mark} |")
        out.append(f"\nGain over the best single model on test: **{rep['gain_vs_best_single']:+.4f}** micro-F1.\n")


def asr_section(out: list[str]) -> None:
    out.append("## ASR (faster-whisper large-v3, FLEURS ta_in test)\n")
    rep = _load("asr_baseline.json")
    if rep:
        variants = [("current default", rep)]
        for label, fname in (("temperature fallback ladder [0..1.0]", "asr_baseline_fallback_ladder.json"),
                             ("condition_on_previous_text=True", "asr_baseline_condprev_on.json")):
            alt = _load(fname)
            if alt and alt is not rep:
                variants.append((label, alt))
        out.append("| setting | clips | audio | device | decoding | WER | CER | median clip WER | RTF | flagged |")
        out.append("|---|---:|---:|---|---|---:|---:|---:|---:|---:|")
        for label, r in variants:
            temp = r.get("temperature", "ladder (library default)")
            dec = f"temp {temp}" if not isinstance(temp, list) else "temp ladder"
            dec += f", cond_prev={'on' if r.get('condition_on_previous_text', True) else 'off'}"
            out.append(
                f"| {label} | {r['n_clips']} | {r['audio_seconds']:.0f}s | {r['device']}/{r['compute_type']} | {dec} "
                f"| **{_pct(r['wer'])}** | **{_pct(r['cer'])}** | {_pct(r.get('median_clip_wer', 0))} "
                f"| {r['rtf']:.2f} | {r.get('n_flagged', '—')} |"
            )
        out.append("")
        flagged = [c for c in rep.get("per_clip", []) if c.get("flags")]
        if flagged:
            out.append("Quality-guard flags (transcript may contain words not spoken):\n")
            out.append("| clip | length | WER | CER | flag |")
            out.append("|---|---:|---:|---:|---|")
            for c in flagged:
                out.append(f"| {c['id']} | {c['duration']:.1f}s | {_pct(c['wer'])} | {_pct(c['cer'])} | {c['quality']} |")
            out.append("")
        out.append("Tamil is agglutinative: one boundary decision changes the word count, so **CER is the "
                   "honest metric** and per-clip WER > 100% is not by itself a hallucination.\n")
    else:
        out.append("_not run_\n")

    cmp_ = _load("asr_compute_types.json")
    if cmp_:
        out.append(f"### Compute-type sweep ({cmp_['n_clips']} clips)\n")
        out.append("| config | WER | CER | RTF |")
        out.append("|---|---:|---:|---:|")
        for label, r in cmp_["results"].items():
            out.append(f"| {label} | {_pct(r['wer'])} | {_pct(r['cer'])} | {r['rtf']:.3f} |")
        out.append("\nRun with the library's temperature-fallback ladder on, so every RTF here carries the "
                   "re-decoding cost (see the profile below); the WER ordering (int8 best) stands.\n")

    sweep = sorted(REPORTS.glob("asr_baseline_rp*.json")) + sorted(REPORTS.glob("asr_baseline_nr*.json"))
    if sweep and rep:
        out.append("### Decoder brakes on the replay loop (finding 6g; same 60 clips, T=0, batched)\n")
        out.append("| setting | WER | CER | RTF | flagged | clip 1916 WER | clip 1721 WER |")
        out.append("|---|---:|---:|---:|---:|---:|---:|")

        def _clip(r, cid):
            for c in r.get("per_clip", []):
                if c["id"] == cid and c.get("flags") is not None and c["file"] in ("12583250098003224463.wav", "16989398024822917306.wav"):
                    return _pct(c["wer"]) + (" ⚑" if c["flags"] else "")
            return "—"

        for label, r in [("default (rp 1.0, nr 0)", rep)] + [(p.stem.replace("asr_baseline_", ""), _load(p.name)) for p in sweep]:
            if not r:
                continue
            label = label.replace("rp", "rp ").replace("_nr", ", nr ").replace("nr", "nr ") if label != "default (rp 1.0, nr 0)" else label
            out.append(f"| {label} | {_pct(r['wer'])} | {_pct(r['cer'])} | {r['rtf']:.2f} | {r.get('n_flagged', '—')} "
                       f"| {_clip(r, '1916')} | {_clip(r, '1721')} |")
        out.append("\n`rp` = CTranslate2 repetition_penalty, `nr` = no_repeat_ngram_size (token n-grams). ⚑ = the "
                   "transcript guard fired. The pipeline default is whichever row PROGRESS.md finding 6g adopted.\n")

    prof = _load("asr_profile.json")
    if prof:
        out.append(f"### Where the time goes ({prof['n_clips']} clips; {prof.get('gpu', '')}; CTranslate2 {prof.get('ctranslate2', '?')})\n")
        out.append("| compute | mode | beam | temp. fallback | batch | audio | RTF | WER |")
        out.append("|---|---|---:|---|---:|---:|---:|---:|")
        for r in prof["rows"]:
            out.append(f"| {r['compute']} | {r['mode']} | {r['beam']} | {'on' if r.get('fallback', True) else 'off'} "
                       f"| {r['batch'] or '—'} | {r['audio_s']:.0f}s | {r['rtf']:.3f} | {_pct(r['wer'])} |")
        out.append("\n`joined-*` rows transcribe the clips concatenated into one complaint-length recording; "
                   "`batched` uses faster-whisper's BatchedInferencePipeline over its VAD chunks. "
                   "Compare within one run only (laptop GPU clocks vary).\n")


def translation_section(out: list[str]) -> None:
    rep = _load("translation_eval.json")
    out.append("## Translation ta→en (feeds the English-only statute classifier)\n")
    if not rep:
        out.append("_not run_\n")
        return
    out.append(f"Gold: {rep.get('gold', '')}. FLoRes is Wikipedia register, not complaints — a "
               "model-comparison signal, not a deployment estimate.\n")
    out.append("| model | backend | n | chrF | BLEU | s/sentence | device |")
    out.append("|---|---|---:|---:|---:|---:|---|")
    for model, r in rep["results"].items():
        out.append(f"| `{model}` | {r['backend']} | {r['n']} | **{r['chrf']:.1f}** | {r['bleu']:.1f} "
                   f"| {r['sec_per_sentence']:.2f} | {r['device']} |")
    out.append("\nFor scale: published FLoRes ta→en is ~50 chrF (NLLB-600M) and ~55–60 (IndicTrans2). "
               "Translation quality bounds statute-ID quality for Tamil input; the element cue scan on the "
               "original text is the safety net under it (PROGRESS.md finding 5c).\n")


def extraction_section(out: list[str]) -> None:
    rep = _load("extraction_rules.json")
    out.append("## Extraction (rule-based floor, per-slot)\n")
    if not rep:
        out.append("_not run_\n")
        return
    out.append(f"Gold: `{Path(rep['gold']).name}`, {rep['n_docs']} hand-authored adversarial documents "
               "(smoke set, not a benchmark).\n")
    out.append("| slot | tp | fp | fn | P | R | F1 |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    for slot, r in rep["slots"].items():
        label = "**micro**" if slot == "_micro" else slot
        out.append(f"| {label} | {r['tp']} | {r['fp']} | {r['fn']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |")
    out.append("\nZero false positives is the property that matters: a wrong value on the form invites "
               "trust; a blank prompts a question.\n")


def mapping_section(out: list[str]) -> None:
    from fir.statute.ipc_bns_map import IpcBnsMap

    cov = IpcBnsMap.load().coverage()
    out.append("## IPC → BNS mapping\n")
    out.append("| rows | auto-applicable | held for review | non-BNS targets |")
    out.append("|---:|---:|---:|---:|")
    out.append(f"| {cov['total']} | {cov['auto_applicable']} | {cov['needs_review']} | {cov['non_bns_targets']} |")
    out.append("\nAuto-apply rule: `confidence == high AND needs_review == N AND target is a BNS section`. "
               "Source is a third-party pocket directory, not the MHA gazette.\n")


def build() -> str:
    out: list[str] = [
        f"# FIR-TN — results as of {date.today().isoformat()}\n",
        "All numbers below are read from `artifacts/reports/*.json`; nothing is recomputed here. "
        "Caveats and interpretation live in `PROGRESS.md`.\n",
    ]
    statute_section(out)
    ensemble_section(out)
    asr_section(out)
    translation_section(out)
    extraction_section(out)
    mapping_section(out)
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()
    md = build()
    if args.stdout:
        print(md)
    else:
        REPORTS.mkdir(parents=True, exist_ok=True)
        (REPORTS / "RESULTS.md").write_text(md, encoding="utf-8")
        print(f"wrote {(REPORTS / 'RESULTS.md').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
