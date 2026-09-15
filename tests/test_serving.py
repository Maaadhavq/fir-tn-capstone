"""The FastAPI surface, driven through TestClient with a stub classifier.

The real TF-IDF artifact is not required: the pipeline is swapped for one built
on the golden stub so these tests run anywhere `make test` does.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os

    from fastapi.testclient import TestClient

    os.environ["FIR_AUDIT_LOG"] = str(tmp_path_factory.mktemp("audit") / "drafts.jsonl")

    from fir.orchestrator.graph import build_slice_graph
    from fir.serving import app as serving
    from fir.statute.ipc_bns_map import IpcBnsMap
    from tests.stubs import GOLDEN_RULES, StubAsr, StubStatuteClassifier

    # inject a pre-built graph so no artifact is needed
    serving._pipe.graph = build_slice_graph(
        classifier=StubStatuteClassifier(rules=GOLDEN_RULES),
        asr=StubAsr(default="the accused stole a mobile phone worth Rs 15,000"),
        ipc_map=IpcBnsMap.load(),
    )
    serving._pipe.classifier_name = "stub-classifier"
    return TestClient(serving.app)


def test_health_answers_without_models(client):
    r = client.get("/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["local_only"] is True
    assert body["files_anything"] is False
    assert body["classifier"] == "stub-classifier"


def test_mapping_coverage_endpoint(client):
    body = client.get("/v1/mapping/coverage").json()
    assert body["total"] == 100
    assert body["auto_applicable"] == 69
    assert "NOT the official MHA gazette" in body["source"]


def test_text_complaint_returns_a_full_draft(client):
    r = client.post(
        "/v1/complaint/text",
        json={"narrative": "On 8.5.2026 the accused stole a mobile phone worth Rs 15,000."},
    )
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["decision"]["route"] == "FIR"
    assert [s["bns_section"] for s in body["decision"]["suggested_sections"]] == ["BNS 303(2)"]

    rec = body["record"]
    assert rec["schema_version"] == "if1/v1"
    assert rec["status"] == "draft"
    assert rec["occurrence"]["from_datetime"]["value"] == "2026-05-08"
    assert rec["total_property_value_inr"]["value"] == 15000.0

    assert "FIRST INFORMATION REPORT" in body["if1_text"]
    assert body["narrative_grounded"] is True
    assert body["asr_flags"] == []
    assert "Not filed" in body["disclaimer"]
    assert body["elapsed_ms"] >= 0


def test_annotate_flag_tags_the_form(client):
    plain = client.post("/v1/complaint/text",
                        json={"narrative": "he stole a phone on 1.1.2026"}).json()["if1_text"]
    tagged = client.post("/v1/complaint/text",
                         json={"narrative": "he stole a phone on 1.1.2026", "annotate": True}).json()["if1_text"]
    assert "⟨" not in plain
    assert "⟨extracted" in tagged


def test_review_case_routes_to_officer(client):
    body = client.post("/v1/complaint/text",
                       json={"narrative": "the accused published defamatory statements"}).json()
    assert body["decision"]["route"] == "OFFICER_REVIEW"
    assert body["decision"]["review_flags"][0]["reason"] == "flagged_needs_review"
    assert body["record"]["acts_sections"][0]["mapping_ambiguous"] is True


def test_empty_narrative_is_rejected(client):
    assert client.post("/v1/complaint/text", json={"narrative": ""}).status_code == 422


def test_server_never_returns_a_verified_record(client):
    for text in ("stole a phone", "harassed for dowry", "slapped him", "nothing"):
        body = client.post("/v1/complaint/text", json={"narrative": text}).json()
        assert body["record"]["status"] != "verified"


def test_audio_endpoint_runs_asr_and_does_not_retain_the_file(client, tmp_path):
    wav = tmp_path / "c.wav"
    wav.write_bytes(b"RIFF....WAVEfmt ")   # the stub ASR never reads it
    with wav.open("rb") as fh:
        r = client.post("/v1/complaint/audio", files={"file": ("c.wav", fh, "audio/wav")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["decision"]["source"] == "audio"
    assert body["decision"]["transcript"] == "the accused stole a mobile phone worth Rs 15,000"
    assert body["record"]["information_type"] == "oral"
    assert body["decision"]["route"] == "FIR"


def test_if1_schema_endpoint_serves_the_canonical_schema(client):
    body = client.get("/v1/schema/if1").json()
    assert body["title"] == "TN FIR (CCTNS IF-1)"
    assert "narrative" in body["properties"]


def test_unbuilt_pipeline_reports_503_with_a_reason():
    """A box without the trained artifact must say so, not 500."""
    from fastapi.testclient import TestClient

    from fir.serving import app as serving

    saved = serving._pipe
    try:
        serving._pipe = serving._Pipeline()
        serving._pipe._classifier = lambda: (_ for _ in ()).throw(
            FileNotFoundError("no trained TF-IDF model at artifacts/models/tfidf_statute.joblib")
        )
        c = TestClient(serving.app)
        r = c.post("/v1/complaint/text", json={"narrative": "x"})
        assert r.status_code == 503
        assert "tfidf_statute.joblib" in r.json()["detail"]
        assert c.get("/v1/health").status_code == 200   # health still answers
    finally:
        serving._pipe = saved


def test_index_serves_the_verification_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Officer verification" in r.text
    # the one control that must never work from this screen
    assert 'id="verify" disabled' in r.text


def test_statute_lookup_returns_gazette_text_and_schedule_entry(client):
    s = client.get("/v1/statute/303(2)").json()
    assert s["found"] and s["heading"] == "Theft."
    assert s["text"].startswith("(2) Whoever commits theft")
    assert s["cognizable"] == "conditional" and "5,000" in s["schedule_note"]
    assert s["bailable"] == "conditional" and s["triable_by"] == "Any Magistrate."
    assert "bnss_schedule1.csv" in s["schedule_source"]

    m = client.get("/v1/statute/BNS%20103(1)").json()
    assert m["cognizable"] == "cognizable" and m["heading"] == "Punishment for murder."
    assert m["bailable"] == "non_bailable" and m["triable_by"] == "Court of Session."

    base = client.get("/v1/statute/80").json()        # printed only as 80(2): derived base
    assert base["found"] and base["cognizable"] == "cognizable" and base["heading"] == "Dowry death."


def test_statute_lookup_unknown_section_is_not_an_error(client):
    r = client.get("/v1/statute/999")
    assert r.status_code == 200
    assert r.json()["found"] is False and r.json()["cognizable"] == "unknown"


def test_browser_recording_blob_is_accepted_by_mime_type(client):
    """MediaRecorder posts `audio/webm;codecs=opus` with whatever filename the
    page gives it; the suffix PyAV needs comes from the MIME type."""
    r = client.post("/v1/complaint/audio",
                    files={"file": ("recording", b"\x1a\x45\xdf\xa3 webm bytes", "audio/webm;codecs=opus")})
    assert r.status_code == 200, r.text
    assert r.json()["decision"]["source"] == "audio"


def test_unknown_audio_type_is_415_and_empty_upload_is_400(client):
    r = client.post("/v1/complaint/audio", files={"file": ("x.exe", b"MZ....", "application/octet-stream")})
    assert r.status_code == 415 and ".webm" in r.json()["detail"]
    r = client.post("/v1/complaint/audio", files={"file": ("c.wav", b"", "audio/wav")})
    assert r.status_code == 400


def test_oversized_audio_is_413(client, monkeypatch):
    from fir.serving import app as serving

    monkeypatch.setattr(serving, "MAX_AUDIO_BYTES", 1024)
    r = client.post("/v1/complaint/audio", files={"file": ("c.wav", b"x" * 4096, "audio/wav")})
    assert r.status_code == 413


def test_health_reports_prewarm_off_and_no_resident_asr_by_default(client, monkeypatch):
    monkeypatch.delenv("FIR_PREWARM", raising=False)
    h = client.get("/v1/health").json()
    assert h["prewarm"] is False and h["asr_loaded"] is False


def test_every_draft_is_audit_logged_without_its_text(client):
    import json
    import os
    from pathlib import Path

    log = Path(os.environ["FIR_AUDIT_LOG"])
    before = len(log.read_text(encoding="utf-8").splitlines()) if log.exists() else 0
    r1 = client.post("/v1/complaint/text", json={"narrative": "harassed for dowry and found dead"}).json()
    r2 = client.post("/v1/complaint/text", json={"narrative": "the neighbour slapped him during an argument"}).json()
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == before + 2                      # append-only, one line per draft
    e1, e2 = json.loads(lines[-2]), json.loads(lines[-1])
    assert e1["record_id"] == r1["record"]["record_id"] and e2["record_id"] == r2["record"]["record_id"]
    assert e1["route"] == "FIR" and "BNS 80" in e1["bns_sections"]
    assert e2["route"] == "CSR" and e2["status"] == "csr_advisory"
    assert "dowry" not in lines[-2] and "slapped" not in lines[-1]   # hashes, not text
    assert len(e1["narrative_sha256"]) == 64
    assert client.get("/v1/health").json()["audit_entries"] == before + 2


def test_audit_log_can_be_disabled(client, monkeypatch):
    monkeypatch.setenv("FIR_AUDIT_LOG", "")
    from fir.serving import audit

    assert audit.log_path() is None and audit.count() == 0
    assert audit.append({"x": 1}) is None
    assert client.post("/v1/complaint/text", json={"narrative": "stole a phone worth Rs 9,000"}).status_code == 200
