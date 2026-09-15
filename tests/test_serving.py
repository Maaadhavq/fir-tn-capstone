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
def client():
    from fastapi.testclient import TestClient

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
