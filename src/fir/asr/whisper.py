"""faster-whisper wrapper for Tamil ASR.

faster-whisper runs Whisper through CTranslate2, which quantises to int8 at load
time from the float16 checkpoint. large-v3 at int8 needs roughly 1.6 GB of VRAM,
comfortably inside this box's 8 GB.

Device selection falls back rather than failing: CTranslate2's CUDA kernels are
compiled per-architecture, and this GPU is Blackwell (sm_120). If the installed
wheel predates sm_120 support, the model load raises, and a 20-clip baseline is
perfectly runnable on CPU int8 -- slower, identical WER. A silent crash on the
first GPU-only box would be much worse than a slow run.
"""

from __future__ import annotations

import ctypes
import glob
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from fir.asr.quality import QualityReport, assess
from fir.config import pipeline

_CUDA_LIBS_PRELOADED = False


def preload_cuda_libs() -> list[str]:
    """Make torch's bundled cuDNN/cuBLAS visible to CTranslate2.

    CTranslate2 `dlopen`s cuDNN by soname at model-load time. The pip `torch`
    wheel ships cuDNN 9 inside `site-packages/nvidia/`, which is not on the
    dynamic loader path, so the lookup fails with:

        Unable to load any of {libcudnn_ops.so.9.1.0, ...}
        Invalid handle. Cannot load symbol cudnnCreateTensorDescriptor

    That failure aborts the process at the C level -- it is NOT a Python
    exception, so a try/except around WhisperModel() cannot catch it and the
    device fallback below never gets a chance to run. Loading the libraries
    ourselves with RTLD_GLOBAL first puts them in the process by soname, and
    CTranslate2's later dlopen resolves against them.

    Setting LD_LIBRARY_PATH would also work but only if exported before the
    interpreter starts, which we cannot rely on. Returns what was loaded.
    """
    global _CUDA_LIBS_PRELOADED
    if _CUDA_LIBS_PRELOADED:
        return []

    loaded: list[str] = []
    try:
        import nvidia

        roots = [Path(p) for p in nvidia.__path__]
    except ImportError:
        roots = []

    # cuBLAS before cuDNN: cuDNN's own dependencies resolve against it.
    for pattern in ("cublas/lib/libcublas*.so*", "cudnn/lib/libcudnn*.so*"):
        for root in roots:
            for path in sorted(glob.glob(str(root / pattern))):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                    loaded.append(os.path.basename(path))
                except OSError:
                    continue

    _CUDA_LIBS_PRELOADED = True
    return loaded


@dataclass(slots=True)
class Transcript:
    """One decoded utterance."""

    text: str
    language: str = ""
    language_probability: float = 0.0
    duration: float = 0.0
    segments: list[dict] = field(default_factory=list)
    quality: "QualityReport | None" = None

    @property
    def flags(self) -> list[str]:
        """Quality flags as plain strings, for the graph state and the report."""
        return [f.value for f in self.quality.flags] if self.quality else []


class WhisperAsr:
    """Lazy-loading faster-whisper wrapper.

    The model is not loaded until the first `transcribe` call, so importing this
    module (as the LangGraph builder does) costs nothing.
    """

    def __init__(
        self,
        model_name: str | None = None,
        compute_type: str | None = None,
        device: str | None = None,
        language: str | None = None,
        beam_size: int | None = None,
        vad_filter: bool | None = None,
        condition_on_previous_text: bool | None = None,
        temperature: float | list[float] | None = None,
        batch_size: int | None = None,
    ):
        cfg = pipeline()["asr"]
        self.model_name = model_name or cfg["model"]
        self.compute_type = compute_type or cfg["compute_type"]
        self.device = device or cfg.get("device", "auto")
        self.language = language or cfg.get("language", "ta")
        self.beam_size = beam_size if beam_size is not None else cfg.get("beam_size", 5)
        self.vad_filter = (
            vad_filter if vad_filter is not None else cfg.get("vad_filter", True)
        )
        # Off by default. With it on, Whisper feeds each 30s window's output back
        # in as the prompt for the next, and a single bad window propagates into
        # the replay loop seen on clip 1916. Off costs a little cross-window
        # coherence and buys a hard stop on that failure mode.
        self.condition_on_previous_text = (
            condition_on_previous_text
            if condition_on_previous_text is not None
            else cfg.get("condition_on_previous_text", False)
        )
        # Single pass at temperature 0 by default. faster-whisper's own default
        # is a *fallback ladder* [0, .2, .4, .6, .8, 1.0]: whenever a window's
        # output fails the compression-ratio or avg-logprob (-1.0) check it is
        # decoded again at the next temperature, up to six times. Low-confidence
        # Tamil fails that check constantly, so the ladder cost 5.3x wall-clock
        # (RTF 3.06 -> 0.57 on 8 FLEURS clips, CTranslate2 4.8.2) *and* the
        # high-temperature passes are sampled, which is where invented text comes
        # from. Measured, not assumed: PROGRESS.md finding 6f.
        self.temperature = temperature if temperature is not None else cfg.get("temperature", 0.0)
        # > 0: decode through faster-whisper's BatchedInferencePipeline, which
        # VAD-splits the recording into speech chunks and decodes them together.
        # On a complaint-length recording that was 3x faster than the sliding
        # 30 s window *and* more accurate (finding 6f: RTF 0.31 -> 0.105, WER
        # 0.81 -> 0.49 at beam 5), because chunks follow the speaker's pauses
        # instead of cutting sentences at fixed offsets. 0 = sequential decode.
        self.batch_size = int(batch_size if batch_size is not None else cfg.get("batch_size", 0))
        self._model = None
        self._batched = None
        self.resolved_device: str = ""
        self.resolved_compute_type: str = ""

    # -- loading ------------------------------------------------------------
    def _candidate_devices(self) -> list[tuple[str, str]]:
        """(device, compute_type) pairs to try, best first."""
        if self.device == "cpu":
            return [("cpu", "int8")]
        if self.device == "cuda":
            return [("cuda", self.compute_type)]

        # auto
        try:
            import torch

            cuda = torch.cuda.is_available()
        except Exception:
            cuda = False
        pairs = []
        if cuda:
            pairs.append(("cuda", self.compute_type))
        pairs.append(("cpu", "int8"))
        return pairs

    def load(self):
        if self._model is not None:
            return self._model

        from faster_whisper import WhisperModel

        preloaded = preload_cuda_libs()
        if preloaded:
            print(f"==> preloaded CUDA libs: {', '.join(preloaded[:4])} ...", flush=True)

        errors: list[str] = []
        for device, compute_type in self._candidate_devices():
            try:
                print(
                    f"==> loading {self.model_name} on {device} ({compute_type})",
                    flush=True,
                )
                self._model = WhisperModel(
                    self.model_name, device=device, compute_type=compute_type
                )
                self.resolved_device = device
                self.resolved_compute_type = compute_type
                return self._model
            except Exception as exc:  # noqa: BLE001 -- we report every attempt
                errors.append(f"{device}/{compute_type}: {type(exc).__name__}: {exc}")
                print(f"    failed on {device}, falling back", flush=True)

        raise RuntimeError(
            "could not load faster-whisper on any device:\n  " + "\n  ".join(errors)
        )

    # -- inference ----------------------------------------------------------
    def transcribe(self, audio_path: str | Path) -> Transcript:
        model = self.load()
        if self.batch_size > 0:
            if self._batched is None:
                from faster_whisper import BatchedInferencePipeline

                self._batched = BatchedInferencePipeline(model)
            # the batched pipeline needs VAD to make its chunks; it has no
            # cross-window prompt, so condition_on_previous_text does not apply
            segments, info = self._batched.transcribe(
                str(audio_path),
                language=self.language,
                beam_size=self.beam_size,
                batch_size=self.batch_size,
                vad_filter=True,
                temperature=self.temperature,
            )
        else:
            segments, info = model.transcribe(
                str(audio_path),
                language=self.language,
                beam_size=self.beam_size,
                vad_filter=self.vad_filter,
                condition_on_previous_text=self.condition_on_previous_text,
                temperature=self.temperature,
            )
        # `segments` is a generator; decoding happens on iteration.
        segs = [
            {"start": s.start, "end": s.end, "text": s.text} for s in segments
        ]
        text = " ".join(s["text"].strip() for s in segs).strip()
        return Transcript(
            text=text,
            language=info.language,
            language_probability=float(info.language_probability),
            duration=float(info.duration),
            segments=segs,
            quality=assess(text),
        )

    def transcribe_batch(
        self, audio_paths: list[str | Path], progress: bool = True
    ) -> tuple[list[Transcript], float]:
        """Transcribe a list of clips; returns transcripts and wall-clock seconds."""
        self.load()
        out: list[Transcript] = []
        t0 = time.perf_counter()
        for i, p in enumerate(audio_paths, 1):
            out.append(self.transcribe(p))
            if progress:
                print(f"    [{i}/{len(audio_paths)}] {Path(p).name}", flush=True)
        return out, time.perf_counter() - t0
