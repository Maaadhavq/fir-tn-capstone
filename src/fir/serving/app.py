"""FastAPI surface over the pipeline graph.

Local-only by design: this binds to localhost, talks to no external service,
and every model it uses is loaded from disk. Two things it deliberately does
not do:

* It never returns a record with `status="verified"` or offers an endpoint that
  would set it. Verification is the officer's act, performed in Stage E, and the
  server has no business doing it.

* It never files anything. There is no CCTNS integration here and there will
  not be one in this codebase.

Models load lazily on first use so `GET /v1/health` answers instantly, and a
box without the trained TF-IDF artifact still serves the health and mapping
endpoints while telling you plainly what is missing.

    uvicorn fir.serving.app:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import functools
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from fir.config import ARTIFACT_DIR
from fir.config import models as models_cfg
from fir.schema.if1 import FirRecord
from fir.schema.statute_decision import StatuteDecision
from fir.statute.ipc_bns_map import IpcBnsMap

app = FastAPI(
    title="FIR-TN decision support",
    version="0.1.0",
    description=(
        "Drafts a TN IF-1 from a Tamil / Tamil-English complaint for officer "
        "verification. Decision support only. Nothing is filed."
    ),
)

_STARTED = time.time()
_STATIC = Path(__file__).resolve().parent / "static"


# ---------------------------------------------------------------------------
# lazily-built pipeline
# ---------------------------------------------------------------------------


class _Pipeline:
    """Holds the compiled graph and the models it needs; builds on first call."""

    def __init__(self, classifier_choice: str = "auto") -> None:
        self.graph = None
        self.classifier_choice = classifier_choice
        self.classifier_name: str | None = None
        self.translator = None
        self.asr_loaded = False
        self.error: str | None = None
        self._asr = None

    def _classifier(self):
        from fir.statute import registry

        return registry.load(self.classifier_choice)

    def _asr_backend(self):
        if self._asr is None:
            from fir.asr.whisper import WhisperAsr

            self._asr = WhisperAsr()
        return self._asr

    def ensure(self, need_asr: bool = False):
        if self.graph is None:
            from fir.orchestrator.graph import build_slice_graph

            try:
                clf = self._classifier()
            except FileNotFoundError as exc:
                self.error = str(exc)
                raise HTTPException(status_code=503, detail=self.error) from exc
            from fir.translate import default_translator

            self.classifier_name = clf.name
            self.translator = default_translator()
            self.graph = build_slice_graph(
                classifier=clf,
                asr=self._asr_backend(),      # lazy inside; costs nothing until used
                ipc_map=IpcBnsMap.load(),
                translator=self.translator,   # None when IndicTrans2 is not cached
            )
        if need_asr:
            self.asr_loaded = True
        return self.graph


_pipe = _Pipeline()


# ---------------------------------------------------------------------------
# request / response models
# ---------------------------------------------------------------------------


class TextComplaint(BaseModel):
    narrative: str = Field(min_length=1, description="complaint text, Tamil / English / mixed")
    annotate: bool = Field(default=False, description="tag machine-filled slots on the form")


class DraftResponse(BaseModel):
    """Everything Stage E needs to render a verification screen."""

    decision: StatuteDecision
    record: FirRecord
    if1_text: str
    narrative_grounded: bool
    narrative_en: str | None = Field(default=None, description="what the classifier read, if translated")
    translation: dict[str, Any] = Field(default_factory=dict)
    asr_flags: list[str] = Field(default_factory=list)
    fill_report: dict[str, Any] = Field(default_factory=dict)
    elapsed_ms: int
    disclaimer: str = (
        "Draft for officer verification. Not filed. The officer is the author of record."
    )


def _run(graph, payload: dict, annotate: bool) -> DraftResponse:
    from fir.instantiate.render import render_if1
    from fir.orchestrator.graph import to_decision

    t0 = time.perf_counter()
    state = graph.invoke({**payload, "errors": []})
    if state.get("errors") and not state.get("fir_record"):
        raise HTTPException(status_code=500, detail="; ".join(state["errors"]))

    record = FirRecord.model_validate(state["fir_record"])
    assert record.status != "verified"  # belt and braces: the server never verifies
    return DraftResponse(
        decision=to_decision(state),
        record=record,
        if1_text=render_if1(record, annotate=annotate) if annotate else state["if1_text"],
        narrative_grounded=bool(state.get("narrative_grounded")),
        narrative_en=state.get("narrative_en") if state.get("translation") else None,
        translation=state.get("translation") or {},
        asr_flags=list(state.get("asr_flags") or []),
        fill_report=state.get("fill_report") or {},
        elapsed_ms=int((time.perf_counter() - t0) * 1000),
    )


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------


@app.get("/v1/health")
def health() -> dict[str, Any]:
    tfidf = (ARTIFACT_DIR / "models" / "tfidf_statute.joblib").exists()
    bert = (ARTIFACT_DIR / "models" / "inlegalbert_statute" / "statute_head.json").exists()
    return {
        "status": "ok",
        "uptime_s": int(time.time() - _STARTED),
        "pipeline_built": _pipe.graph is not None,
        "classifier": _pipe.classifier_name,
        "asr_loaded": _pipe.asr_loaded,
        "artifacts": {"tfidf_statute": tfidf, "inlegalbert_statute": bert},
        "classifier_choice": _pipe.classifier_choice,
        "translator": getattr(_pipe.translator, "model_id", None),
        "asr_model": models_cfg()["asr"]["primary"]["name"],
        "local_only": True,
        "files_anything": False,
    }


@app.get("/v1/mapping/coverage")
def mapping_coverage() -> dict[str, Any]:
    m = IpcBnsMap.load()
    return {
        **m.coverage(),
        "auto_apply_rule": "confidence == high AND needs_review == N AND target is a BNS section",
        "source": "third-party BNS-IPC pocket directory -- NOT the official MHA gazette",
    }


class StatuteLookup(BaseModel):
    """One BNS section as the officer would look it up: the words, and what
    the First Schedule says about it. Read-only reference; not a decision."""

    bns_section: str
    found: bool
    heading: str | None = None
    chapter: str | None = None
    chapter_title: str | None = None
    text: str | None = Field(default=None, description="operative text of the section or named sub-section")
    gazette_page: int | None = None
    cognizable: str = "unknown"
    schedule_note: str = ""
    bailable: str | None = None
    triable_by: str | None = None
    schedule_source: str = ""


@app.get("/v1/statute/{bns_section}", response_model=StatuteLookup)
def statute_lookup(bns_section: str) -> StatuteLookup:
    """Look up a BNS section: gazette text + First Schedule classification.

    `bns_section` may be "303", "303(2)" or "BNS 303(2)". Composite targets
    ("324(4),(5)") resolve the way routing does: only if the parts agree."""
    from fir.statute.bns_text import bns_text
    from fir.statute.cognizability import schedule, schedule_source

    key = bns_section.strip()
    bt = bns_text()
    sec = bt.section(key) if bt else None
    sch = schedule()
    norm = sch._normalise(key)
    row_key = norm if norm in sch.table else sch._key_for(norm)
    extra = _schedule_row(row_key) if row_key else {}
    return StatuteLookup(
        bns_section=key,
        found=sec is not None or row_key is not None,
        heading=sec.heading if sec else None,
        chapter=sec.chapter if sec else None,
        chapter_title=sec.chapter_title if sec else None,
        text=bt.text_for(key) if (bt and sec) else None,
        gazette_page=sec.gazette_page if sec else None,
        cognizable=sch.lookup(key).value,
        schedule_note=sch.note_for(key),
        bailable=extra.get("bailable"),
        triable_by=extra.get("triable_by"),
        schedule_source=schedule_source(),
    )


@functools.lru_cache(maxsize=1)
def _schedule_rows() -> dict[str, dict]:
    """The Schedule CSV's extra columns (bailable, court), keyed like the loader keys."""
    import csv

    from fir.config import DATA_DIR

    p = DATA_DIR / "statutes" / "bnss_schedule1.csv"
    if not p.exists():
        return {}
    with p.open(encoding="utf-8", newline="") as fh:
        return {row["bns_section"].strip().upper(): row for row in csv.DictReader(fh)}


def _schedule_row(key: str) -> dict:
    return _schedule_rows().get(key.upper(), {})


@app.post("/v1/complaint/text", response_model=DraftResponse)
def complaint_text(body: TextComplaint) -> DraftResponse:
    graph = _pipe.ensure()
    return _run(graph, {"narrative": body.narrative}, body.annotate)


@app.post("/v1/complaint/audio", response_model=DraftResponse)
async def complaint_audio(
    file: UploadFile = File(..., description="wav/mp3/m4a, Tamil or Tamil-English"),
    annotate: bool = False,
) -> DraftResponse:
    graph = _pipe.ensure(need_asr=True)
    suffix = Path(file.filename or "clip.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    try:
        return _run(graph, {"audio_path": str(tmp_path)}, annotate)
    finally:
        tmp_path.unlink(missing_ok=True)   # the audio is not retained


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Stage E v0: the officer verification screen, a single static page that
    drives the API. A React app replaces it later; the API contract does not
    change."""
    return FileResponse(_STATIC / "index.html", media_type="text/html")


@app.get("/v1/schema/if1")
def if1_schema() -> dict[str, Any]:
    """The canonical IF-1 JSON Schema, for Stage E form generation."""
    import json

    return json.loads(
        (Path(__file__).resolve().parents[1] / "schema" / "if1_v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
