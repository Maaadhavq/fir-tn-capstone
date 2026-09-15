"""ILSI loader (Zenodo 10.5281/zenodo.6053791).

Line format, verified against the real corpus:

    {"id": "100002997",
     "text": ["sentence one.", "sentence two.", ...],
     "labels": ["Section 302 in The Indian Penal Code", ...]}

Two deliberate choices:

1. **Streaming.** `train.jsonl` is 320 MB and WSL gets ~7.7 GB of RAM by
   default. `iter_ilsi` yields one record at a time so callers that only need
   text+labels never materialise the intermediate JSON objects.

2. **No HuggingFace `datasets`.** It pulls in pyarrow, which Smart App Control
   blocks on the Windows host. Plain `json` handles JSONL fine.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from fir.config import DATA_DIR, datasets

# "Section 304B in The Indian Penal Code" -> "304B"
_LABEL_RE = re.compile(r"Section\s+([0-9]+[A-Z]*)\s+in\s+The\s+Indian\s+Penal\s+Code", re.I)

SPLITS = ("train", "dev", "test")


def parse_ipc_section(label: str) -> str | None:
    """Extract the bare section number from an ILSI label string.

    Returns None for labels that do not match the expected pattern, rather than
    guessing -- an unparseable label is a data problem we want surfaced.
    """
    m = _LABEL_RE.search(label)
    return m.group(1).upper() if m else None


@dataclass(slots=True)
class FactInstance:
    """One ILSI case: the fact statement plus its gold IPC sections."""

    id: str
    text: str
    ipc_labels: list[str] = field(default_factory=list)

    @property
    def ipc_sections(self) -> list[str]:
        """Gold labels reduced to bare section numbers, e.g. ['302', '498A']."""
        out = [parse_ipc_section(lbl) for lbl in self.ipc_labels]
        return [s for s in out if s]


def _split_path(split: str) -> Path:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    cfg = datasets()["ilsi"]
    return DATA_DIR / "ilsi" / cfg["files"][split]


def iter_ilsi(split: str, limit: int | None = None) -> Iterator[FactInstance]:
    """Stream ILSI records for `split`, at most `limit` of them."""
    path = _split_path(split)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: bash scripts/download_ilsi.sh"
        )
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh):
            if limit is not None and n >= limit:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = rec["text"]
            yield FactInstance(
                id=str(rec["id"]),
                text=" ".join(text) if isinstance(text, list) else str(text),
                ipc_labels=list(rec.get("labels", [])),
            )


def load_ilsi(split: str, limit: int | None = None) -> list[FactInstance]:
    """Eager version of `iter_ilsi`."""
    return list(iter_ilsi(split, limit=limit))


def load_label_vocab() -> list[str]:
    """The 100 IPC labels ILSI uses, in their canonical index order.

    Uses the copy under data/mapping/ (committed) so the label space is stable
    even before the 512 MB Zenodo download has landed.
    """
    for candidate in (
        DATA_DIR / "mapping" / "ilsi_ipc_labels.json",
        DATA_DIR / "ilsi" / datasets()["ilsi"]["files"]["label_vocab"],
    ):
        if candidate.exists():
            vocab: dict[str, int] = json.loads(candidate.read_text(encoding="utf-8"))
            return [lbl for lbl, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    raise FileNotFoundError("no ILSI label vocabulary found under data/")


def load_statute_texts() -> dict[str, str]:
    """`secs.jsonl` -> {"Section 302 in The Indian Penal Code": "<text>"}.

    Used later by the RAG / element-verification stage; loaded here so the data
    layer stays in one place.
    """
    path = DATA_DIR / "ilsi" / datasets()["ilsi"]["files"]["statutes"]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: bash scripts/download_ilsi.sh")
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            text = rec.get("text", "")
            out[rec["id"]] = " ".join(text) if isinstance(text, list) else str(text)
    return out
