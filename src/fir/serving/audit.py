"""Append-only log of every draft the service produced (milestone P5.3, v0).

One JSON line per draft, written under a lock, never rewritten: what was
decided, when, from which kind of input, and a hash of the narrative -- not the
narrative itself, and never the audio (PII minimisation; the record the officer
edits lives in CCTNS, not here). Enough to replay "what did the system suggest
for this complaint on that day" against a `record_id`, which is the audit
question a verification workflow has to answer.

Path: `FIR_AUDIT_LOG` (default artifacts/audit/drafts.jsonl). Set it to an
empty string to disable.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from fir.config import ARTIFACT_DIR

_LOCK = threading.Lock()
_DEFAULT = ARTIFACT_DIR / "audit" / "drafts.jsonl"


def log_path() -> Path | None:
    env = os.environ.get("FIR_AUDIT_LOG")
    if env is None:
        return _DEFAULT
    return Path(env) if env.strip() else None


def narrative_digest(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def append(entry: dict) -> Path | None:
    """Write one line. Returns the path written, or None when disabled."""
    path = log_path()
    if path is None:
        return None
    line = json.dumps({"logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **entry},
                      ensure_ascii=False, sort_keys=True)
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    return path


def count() -> int:
    path = log_path()
    if path is None or not path.exists():
        return 0
    with path.open("rb") as fh:
        return sum(1 for _ in fh)


def entry_for(response) -> dict:
    """The audit view of a DraftResponse: decision, provenance, no free text."""
    d = response.decision
    return {
        "record_id": response.record.record_id,
        "schema_version": response.record.schema_version,
        "source": d.source,
        "classifier": d.classifier,
        "route": d.route,
        "bns_sections": [s.bns_section for s in d.suggested_sections],
        "n_review_flags": len(d.review_flags),
        "n_cue_hits": len(d.cue_hits),
        "asr_flags": list(response.asr_flags),
        "narrative_grounded": response.narrative_grounded,
        "narrative_sha256": narrative_digest(d.narrative),
        "elapsed_ms": response.elapsed_ms,
        "status": response.record.status,
    }
