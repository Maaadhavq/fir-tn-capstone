"""Compare faster-whisper compute types on FLEURS Tamil.

Motivated by a real result: the first baseline run (large-v3, int8, cuda) scored
WER 62.12% at RTF 3.467. Both numbers are wrong-looking in the same direction --
published FLEURS-ta WER for large-v3 sits far lower, and RTF above 1.0 means a
GPU decoding slower than real time. Two of the sample hypotheses also showed
Hebrew characters and early truncation, which is what Whisper does when its
numerics degrade rather than when the audio is hard.

So the question this script answers is: is int8 on this GPU (Blackwell sm_120,
CTranslate2 4.5.0) costing us accuracy and speed, and does float16 fix it?

    python -m harness.compare_asr_compute_types --n 10

Also sweeps VAD, because an over-eager voice-activity filter truncates clips and
would produce the same symptom from a completely different cause.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fir.asr.whisper import WhisperAsr  # noqa: E402
from fir.config import ARTIFACT_DIR  # noqa: E402
from fir.data.fleurs import load_fleurs  # noqa: E402
from harness.metrics import score_asr  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--configs",
        default="int8:vad,float16:vad,float16:novad,int8_float16:vad",
        help="comma-separated compute_type:vad|novad pairs",
    )
    args = ap.parse_args()

    clips = load_fleurs(args.split, limit=args.n)
    if not clips:
        print("no clips found", file=sys.stderr)
        return 1
    audio_seconds = sum(c.duration_seconds for c in clips)
    refs = [c.transcription for c in clips]
    print(f"==> {len(clips)} clips, {audio_seconds:.1f}s of audio\n")

    rows = []
    for spec in args.configs.split(","):
        compute_type, _, vad_spec = spec.partition(":")
        vad = vad_spec != "novad"
        label = f"{compute_type}/{'vad' if vad else 'no-vad'}"
        print(f"==> {label}")
        try:
            asr = WhisperAsr(
                device=args.device, compute_type=compute_type, vad_filter=vad
            )
            hyps, elapsed = asr.transcribe_batch(
                [c.audio_path for c in clips], progress=False
            )
            sc = score_asr(refs, [h.text for h in hyps], audio_seconds, elapsed)
            rows.append((label, sc, asr.resolved_device))
            print(
                f"    WER {sc.wer * 100:6.2f}%   CER {sc.cer * 100:6.2f}%   "
                f"RTF {sc.rtf:.3f}   ({elapsed:.0f}s on {asr.resolved_device})\n"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"    FAILED: {type(exc).__name__}: {exc}\n")

    print("=" * 66)
    print(f"{'config':<22s} {'WER':>9s} {'CER':>9s} {'RTF':>9s}")
    print("-" * 66)
    for label, sc, dev in rows:
        print(f"{label:<22s} {sc.wer * 100:>8.2f}% {sc.cer * 100:>8.2f}% {sc.rtf:>9.3f}")

    if rows:
        best = min(rows, key=lambda r: r[1].wer)
        print(f"\nbest WER: {best[0]}")

    out = ARTIFACT_DIR / "reports" / "asr_compute_types.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "n_clips": len(clips),
                "audio_seconds": audio_seconds,
                "results": {
                    label: {"wer": sc.wer, "cer": sc.cer, "rtf": sc.rtf, "device": dev}
                    for label, sc, dev in rows
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
