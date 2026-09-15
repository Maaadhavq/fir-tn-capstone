"""ASR baseline: faster-whisper large-v3 (int8) on FLEURS Tamil.

    python -m harness.run_asr_baseline               # 20 clips
    python -m harness.run_asr_baseline --n 100       # more clips
    python -m harness.run_asr_baseline --device cpu  # force CPU int8

Reports corpus-level WER and CER (jiwer) after Tamil normalisation, plus the
real-time factor -- the number that says whether this fits the latency budget
for a live complaint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fir.asr.whisper import WhisperAsr  # noqa: E402
from fir.config import ARTIFACT_DIR, pipeline  # noqa: E402
from fir.data.fleurs import load_fleurs  # noqa: E402
from harness.metrics import normalise_tamil, score_asr  # noqa: E402


def main() -> int:
    cfg = pipeline()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=cfg["slice"]["n_asr_clips"])
    ap.add_argument("--split", default=cfg["asr"]["split"])
    ap.add_argument("--device", default=cfg["asr"].get("device", "auto"))
    ap.add_argument("--compute-type", default=cfg["asr"]["compute_type"])
    ap.add_argument("--show", type=int, default=3, help="example transcripts to print")
    ap.add_argument("--fallback-ladder", action="store_true",
                    help="use faster-whisper's default temperature fallback [0..1.0] instead of one pass at 0")
    ap.add_argument("--tag", default="", help="suffix for the report file name")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="0 = sequential sliding window; >0 = BatchedInferencePipeline (config default)")
    args = ap.parse_args()

    print(f"==> loading FLEURS ta_in/{args.split}")
    clips = load_fleurs(args.split, limit=args.n)
    if not clips:
        print("no clips found. Run: python scripts/download_fleurs.py", file=sys.stderr)
        return 1
    audio_seconds = sum(c.duration_seconds for c in clips)
    print(f"    {len(clips)} clips, {audio_seconds:.1f}s of audio")

    asr = WhisperAsr(device=args.device, compute_type=args.compute_type,
                     temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0] if args.fallback_ladder else None,
                     batch_size=args.batch_size)
    transcripts, elapsed = asr.transcribe_batch([c.audio_path for c in clips])

    scores = score_asr(
        references=[c.transcription for c in clips],
        hypotheses=[t.text for t in transcripts],
        audio_seconds=audio_seconds,
        elapsed_seconds=elapsed,
    )

    print("\n" + "=" * 78)
    print("ASR BASELINE -- FLEURS ta_in")
    print("=" * 78)
    print(
        scores.as_report(
            f"{asr.model_name} [{asr.resolved_device}/{asr.resolved_compute_type}]"
        )
    )
    print(f"  wall clock   : {elapsed:.1f}s")

    for c, t in list(zip(clips, transcripts))[: args.show]:
        print(f"\n  --- {c.id} ---")
        print(f"  ref : {normalise_tamil(c.transcription)[:110]}")
        print(f"  hyp : {normalise_tamil(t.text)[:110]}")

    # Per-clip rates, sorted worst-first. Corpus WER cannot distinguish
    # "uniformly mediocre" from "mostly fine, a few catastrophic clips", and
    # those two call for completely different fixes.
    import jiwer

    per_clip = []
    for c, t in zip(clips, transcripts):
        ref, hyp = normalise_tamil(c.transcription), normalise_tamil(t.text)
        if not ref:
            continue
        per_clip.append(
            {
                "id": c.id,
                "file": c.audio_path.name,
                "wer": float(jiwer.wer(ref, hyp)),
                "cer": float(jiwer.cer(ref, hyp)),
                "duration": c.duration_seconds,
                "flags": t.flags,
                "quality": t.quality.summary() if t.quality else "",
                "ref": ref,
                "hyp": hyp,
            }
        )
    per_clip.sort(key=lambda r: -r["wer"])

    flagged = [r for r in per_clip if r["flags"]]
    print(f"\n  quality flags: {len(flagged)} of {len(per_clip)} clips")
    for r in flagged:
        print(f"    {r['id']:>8s}  {r['duration']:5.1f}s  WER {r['wer'] * 100:5.0f}%  {r['quality']}")

    print("\n  worst clips by WER:")
    for r in per_clip[:5]:
        print(f"    {r['wer'] * 100:6.1f}%  {r['id']:>8s}  {r['duration']:5.1f}s  {r['hyp'][:60]}")
    median = per_clip[len(per_clip) // 2]["wer"] if per_clip else 0.0
    print(f"\n  median clip WER: {median * 100:.2f}%   (corpus WER {scores.wer * 100:.2f}%)")
    if median < scores.wer * 0.6:
        print("  -> corpus WER is dominated by a few outlier clips, not a uniform error rate.")

    out = ARTIFACT_DIR / "reports" / f"asr_baseline{('_' + args.tag) if args.tag else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "model": asr.model_name,
                "device": asr.resolved_device,
                "compute_type": asr.resolved_compute_type,
                "split": args.split,
                "n_clips": scores.n_clips,
                "wer": scores.wer,
                "cer": scores.cer,
                "median_clip_wer": median,
                "n_flagged": len(flagged),
                "condition_on_previous_text": asr.condition_on_previous_text,
                "temperature": asr.temperature,
                "batch_size": asr.batch_size,
                "audio_seconds": scores.audio_seconds,
                "elapsed_seconds": elapsed,
                "rtf": scores.rtf,
                "per_clip": per_clip,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nreport -> {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
