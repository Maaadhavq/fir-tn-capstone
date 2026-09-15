"""Where does ASR time go? RTF 1.8-4.2 on a GPU for large-v3 is far too slow.

Times model load, then transcription of a few FLEURS clips under each setting,
and the same audio joined into one complaint-length recording (~2 min) both
sequentially and through faster-whisper's BatchedInferencePipeline. Prints a
table and writes artifacts/reports/asr_profile.json. Nothing here touches the
pipeline defaults -- it is a measurement.

    python -m harness.profile_asr --n 8 --compute int8 float16 --beam 1 5

Caveat to record with the numbers: a laptop GPU on battery downclocks hard, so
compare settings *within* one run, not across runs on different days.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fir.asr.whisper import preload_cuda_libs  # noqa: E402
from fir.config import ARTIFACT_DIR  # noqa: E402
from fir.data.fleurs import load_fleurs  # noqa: E402
from harness.metrics import normalise_tamil  # noqa: E402

OUT = ARTIFACT_DIR / "reports" / "asr_profile.json"


def _wer(ref: str, hyp: str) -> float:
    import jiwer

    ref, hyp = normalise_tamil(ref), normalise_tamil(hyp)
    return float(jiwer.wer(ref, hyp)) if ref else 0.0


def _load_audio(paths) -> tuple[np.ndarray, int]:
    import soundfile as sf

    chunks, sr = [], None
    for p in paths:
        x, r = sf.read(str(p), dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        sr = sr or r
        assert r == sr
        chunks.append(x)
        chunks.append(np.zeros(int(0.4 * sr), dtype="float32"))     # a breath between sentences
    return np.concatenate(chunks), sr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--compute", nargs="+", default=["int8", "float16"])
    ap.add_argument("--beam", nargs="+", type=int, default=[1, 5])
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[4, 8, 16])
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    from faster_whisper import BatchedInferencePipeline, WhisperModel

    from fir.config import pipeline

    cfg = pipeline()["asr"]
    model_name = args.model or cfg["model"]
    clips = load_fleurs("test", limit=args.n)
    if not clips:
        print("no FLEURS clips; run scripts/download_fleurs.py", file=sys.stderr)
        return 1
    audio_s = sum(c.duration_seconds for c in clips)
    joined, sr = _load_audio([c.audio_path for c in clips])
    joined_ref = " ".join(c.transcription for c in clips)
    print(f"{len(clips)} clips, {audio_s:.1f}s audio; joined recording {len(joined) / sr:.1f}s")
    preload_cuda_libs()

    import torch

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    rows = []
    for ct in args.compute:
        t0 = time.perf_counter()
        model = WhisperModel(model_name, device="cuda", compute_type=ct)
        load_s = time.perf_counter() - t0
        print(f"\n== {ct}: model load {load_s:.1f}s")
        # warm-up: the first call pays for kernel selection / cuDNN autotune
        list(model.transcribe(str(clips[0].audio_path), language="ta", beam_size=1, vad_filter=False)[0])

        # The temperature-fallback dimension. faster-whisper's default retries every
        # segment at up to five higher temperatures whenever the greedy/beam pass
        # fails its compression-ratio or avg-logprob check (-1.0). Low-confidence
        # Tamil fails that check often, so each clip may be decoded up to six times
        # -- and the high-temperature *sampled* passes are where fabricated text
        # can come from. `fallback=False` decodes once, at temperature 0.
        for beam in args.beam:
            for fallback in (True, False):
                kw = {} if fallback else {"temperature": 0.0, "compression_ratio_threshold": None,
                                          "log_prob_threshold": None}
                t0 = time.perf_counter()
                hyps = []
                for c in clips:
                    segs, _ = model.transcribe(str(c.audio_path), language="ta", beam_size=beam,
                                               vad_filter=True, condition_on_previous_text=False, **kw)
                    hyps.append(" ".join(s.text.strip() for s in segs))
                el = time.perf_counter() - t0
                wer = float(np.mean([_wer(c.transcription, h) for c, h in zip(clips, hyps)]))
                rows.append({"compute": ct, "mode": "per-clip", "beam": beam, "vad": True, "fallback": fallback,
                             "batch": 0, "audio_s": audio_s, "elapsed_s": el, "rtf": el / audio_s, "wer": wer})
                print(f"  per-clip  beam={beam} fallback={fallback!s:5}       RTF {el / audio_s:.3f}  WER {wer:.3f}",
                      flush=True)

        # complaint-length recording: sequential vs batched, no fallback (the
        # batched pipeline decodes once by design, so compare like with like)
        nofb = {"temperature": 0.0, "compression_ratio_threshold": None, "log_prob_threshold": None}
        for beam in args.beam:
            t0 = time.perf_counter()
            segs, _ = model.transcribe(joined, language="ta", beam_size=beam, vad_filter=True,
                                       condition_on_previous_text=False, **nofb)
            hyp = " ".join(s.text.strip() for s in segs)
            el = time.perf_counter() - t0
            dur = len(joined) / sr
            rows.append({"compute": ct, "mode": "joined-sequential", "beam": beam, "vad": True, "fallback": False,
                         "batch": 0, "audio_s": dur, "elapsed_s": el, "rtf": el / dur, "wer": _wer(joined_ref, hyp)})
            print(f"  joined    beam={beam} sequential            RTF {el / dur:.3f}  WER {_wer(joined_ref, hyp):.3f}",
                  flush=True)
            batched = BatchedInferencePipeline(model)
            for bs in args.batch_sizes:
                t0 = time.perf_counter()
                segs, _ = batched.transcribe(joined, language="ta", beam_size=beam, batch_size=bs,
                                             vad_filter=True, condition_on_previous_text=False, **nofb)
                hyp = " ".join(s.text.strip() for s in segs)
                el = time.perf_counter() - t0
                rows.append({"compute": ct, "mode": "joined-batched", "beam": beam, "vad": True, "fallback": False,
                             "batch": bs, "audio_s": dur, "elapsed_s": el, "rtf": el / dur, "wer": _wer(joined_ref, hyp)})
                print(f"  joined    beam={beam} batched bs={bs:<3}         RTF {el / dur:.3f}  WER {_wer(joined_ref, hyp):.3f}",
                      flush=True)
        del model
        torch.cuda.empty_cache()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    import ctranslate2

    OUT.write_text(json.dumps({"model": model_name, "gpu": gpu, "n_clips": len(clips),
                               "ctranslate2": ctranslate2.__version__, "rows": rows}, indent=2),
                   encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
