"""The `python -m fir` CLI, with the pipeline swapped for the golden stub."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def stub_pipeline(monkeypatch):
    from fir import __main__ as cli
    from fir.orchestrator.graph import build_slice_graph
    from fir.statute.ipc_bns_map import IpcBnsMap
    from tests.stubs import GOLDEN_RULES, StubAsr, StubStatuteClassifier

    def fake(need_asr: bool, classifier: str = "auto"):
        return build_slice_graph(
            classifier=StubStatuteClassifier(rules=GOLDEN_RULES),
            asr=StubAsr(default="the accused stole a mobile phone worth Rs 15,000") if need_asr else None,
            ipc_map=IpcBnsMap.load(),
        )

    monkeypatch.setattr(cli, "_pipeline", fake)
    return cli


def test_draft_prints_result_and_form(stub_pipeline, capsys):
    rc = stub_pipeline.main(["draft", "On 8.5.2026 the accused stole a phone worth Rs 15,000."])
    out = capsys.readouterr().out
    assert rc == 0
    assert "BNS applied  : BNS 303(2)" in out
    assert "faithfulness : PASS" in out
    assert "FIRST INFORMATION REPORT" in out
    assert "13. Action taken : ________" in out


def test_draft_json_is_machine_readable(stub_pipeline, capsys):
    rc = stub_pipeline.main(["draft", "--json", "he stole a phone worth Rs 8,000 on 1.1.2026"])
    assert rc == 0
    body = json.loads(capsys.readouterr().out)
    assert body["decision"]["route"] == "FIR"
    assert body["record"]["schema_version"] == "if1/v1"
    assert body["record"]["occurrence"]["from_datetime"]["value"] == "2026-01-01"
    assert body["narrative_grounded"] is True


def test_draft_annotate_tags_slots(stub_pipeline, capsys):
    stub_pipeline.main(["draft", "--annotate", "he stole a phone on 1.1.2026"])
    assert "⟨extracted" in capsys.readouterr().out


def test_draft_audio_runs_asr(stub_pipeline, capsys, tmp_path):
    wav = tmp_path / "c.wav"
    wav.write_bytes(b"\0" * 16)
    rc = stub_pipeline.main(["draft", "--audio", str(wav)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "transcript   : the accused stole a mobile phone worth Rs 15,000" in out
    assert "BNS 303(2)" in out


def test_draft_with_no_text_errors(stub_pipeline, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(""))
    rc = stub_pipeline.main(["draft"])
    assert rc == 2
    assert "no complaint text" in capsys.readouterr().err


def test_mapping_lookup(capsys):
    from fir import __main__ as cli

    rc = cli.main(["mapping", "302", "500", "161"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "IPC 302    -> BNS 103(1)     AUTO" in out
    assert "IPC 500    -> BNS 356(2)     REVIEW flagged_needs_review" in out
    assert "IPC 161    -> PC Act 1988    REVIEW maps_to_other_statute" in out


def test_mapping_coverage_when_no_sections(capsys):
    from fir import __main__ as cli

    cli.main(["mapping"])
    body = json.loads(capsys.readouterr().out)
    assert body["total"] == 100 and body["auto_applicable"] == 69


def test_check_reports_the_gazette_schedule(capsys):
    from fir import __main__ as cli

    cli.main(["check"])
    out = capsys.readouterr().out
    assert "BNSS Schedule 1" in out
    assert "rows from bnss_schedule1.csv" in out and "stub (" not in out
    assert "IPC->BNS map" in out
