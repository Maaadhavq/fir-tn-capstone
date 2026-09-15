"""FLEURS Tamil (ta_in) loader for the ASR baseline.

FLEURS ships headerless TSVs alongside a tar of 16 kHz wavs. Columns, positional,
verified against the real ta_in file (there is NO speaker_id column, despite
what several dataset cards suggest):

    0 id
    1 file_name
    2 raw_transcription
    3 transcription          <- what we score against
    4 grapheme_transcription  (spaced characters, '|' as word separator)
    5 num_samples
    6 gender

We score against `transcription`, the normalised form, and keep
`raw_transcription` for later error analysis -- it retains the numerals and
punctuation that the ITN stage will eventually have to reproduce.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from fir.config import DATA_DIR

CONFIG = "ta_in"

_ID, _FILE, _RAW, _NORM, _GRAPHEME, _NSAMP, _GENDER = range(7)
_SAMPLE_RATE = 16_000


@dataclass(slots=True)
class AudioClip:
    """One FLEURS utterance."""

    id: str
    audio_path: Path
    transcription: str
    raw_transcription: str = ""
    num_samples: int = 0
    gender: str = ""

    @property
    def duration_seconds(self) -> float:
        return self.num_samples / _SAMPLE_RATE if self.num_samples else 0.0


def _root(split: str) -> tuple[Path, Path]:
    base = DATA_DIR / "fleurs" / CONFIG
    return base / f"{split}.tsv", base / "audio" / split


def iter_fleurs(split: str = "test", limit: int | None = None) -> Iterator[AudioClip]:
    """Stream FLEURS clips whose audio is actually present on disk.

    Rows whose wav is missing are skipped silently -- FLEURS TSVs list more
    utterances than some archives contain, and a partial extraction should
    degrade the clip count rather than crash the run.
    """
    tsv_path, audio_dir = _root(split)
    if not tsv_path.exists():
        raise FileNotFoundError(
            f"{tsv_path} not found. Run: python scripts/download_fleurs.py --split {split}"
        )
    if not audio_dir.exists():
        raise FileNotFoundError(
            f"{audio_dir} not found. Run: python scripts/download_fleurs.py --split {split}"
        )

    n = 0
    with tsv_path.open(encoding="utf-8", newline="") as fh:
        for row in csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            if limit is not None and n >= limit:
                return
            if len(row) <= _NORM:
                continue
            wav = audio_dir / row[_FILE]
            if not wav.exists():
                continue
            yield AudioClip(
                id=row[_ID],
                audio_path=wav,
                transcription=row[_NORM].strip(),
                raw_transcription=row[_RAW].strip(),
                num_samples=int(row[_NSAMP])
                if len(row) > _NSAMP and row[_NSAMP].isdigit()
                else 0,
                gender=row[_GENDER] if len(row) > _GENDER else "",
            )
            n += 1


def load_fleurs(split: str = "test", limit: int | None = None) -> list[AudioClip]:
    return list(iter_fleurs(split, limit=limit))


def is_available(split: str = "test") -> bool:
    tsv_path, audio_dir = _root(split)
    return tsv_path.exists() and audio_dir.exists() and any(audio_dir.glob("*.wav"))
