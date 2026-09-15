"""A browser MediaRecorder blob (webm/opus) must decode without ffmpeg.

faster-whisper decodes through PyAV; the wheel bundles the opus decoder and the
Matroska/WebM demuxer, so the officer page can post what the browser records.
This synthesises one second of a 440 Hz tone with PyAV's libopus encoder and
checks decode_audio returns 16 kHz float samples of the right length.
"""

from __future__ import annotations

import math

import pytest

av = pytest.importorskip("av")
fw_audio = pytest.importorskip("faster_whisper.audio")


def _write_webm(path, seconds: float = 1.0, sr: int = 48000) -> None:
    import numpy as np

    n = int(seconds * sr)
    tone = (0.3 * np.sin(2 * math.pi * 440 * np.arange(n) / sr)).astype("float32")
    with av.open(str(path), "w", format="webm") as out:
        stream = out.add_stream("libopus", rate=sr)
        stream.layout = "mono"
        frame = av.AudioFrame.from_ndarray(tone.reshape(1, -1), format="flt", layout="mono")
        frame.sample_rate = sr
        for packet in stream.encode(frame):
            out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)


def test_webm_opus_decodes_to_16k_float(tmp_path):
    p = tmp_path / "recording.webm"
    _write_webm(p)
    assert p.stat().st_size > 0
    audio = fw_audio.decode_audio(str(p), sampling_rate=16000)
    assert audio.dtype.kind == "f"
    assert abs(len(audio) - 16000) < 1600            # within 10% of one second (opus pre-skip)
    assert float(abs(audio).max()) > 0.05            # not silence
