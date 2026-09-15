"""Golden-regression test for the vertical slice.

Drives the real LangGraph graph, the real IPC->BNS mapping table, and the real
cognizability logic. Only the statute classifier is stubbed -- see tests/stubs.py
for why.

    pytest tests/ -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from fir.statute.cognizability import (  # noqa: E402
    Cognizability,
    Route,
    classify_section,
    route_case,
)
from fir.statute.ipc_bns_map import IpcBnsMap, ReviewReason  # noqa: E402
from tests.stubs import GOLDEN_RULES, StubAsr, StubStatuteClassifier  # noqa: E402

FIXTURE = json.loads(
    (REPO_ROOT / "tests" / "fixtures" / "golden_slice.json").read_text(encoding="utf-8")
)
CASES = FIXTURE["cases"]


@pytest.fixture(scope="module")
def graph():
    from fir.orchestrator.graph import build_slice_graph

    return build_slice_graph(
        classifier=StubStatuteClassifier(rules=GOLDEN_RULES),
        asr=StubAsr(default="the accused stole a mobile phone worth Rs 15,000"),
        ipc_map=IpcBnsMap.load(),
    )


@pytest.fixture(scope="module")
def ipc_map():
    return IpcBnsMap.load()


# ---------------------------------------------------------------------------
# the golden cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_golden_slice(graph, case):
    """End-to-end: narrative -> IPC -> BNS -> route, pinned against the fixture."""
    from fir.orchestrator.graph import run_text

    state = run_text(graph, case["narrative"])
    want = case["expect"]

    assert state.get("predicted_ipc", []) == want["predicted_ipc"], case["why"]
    assert state.get("bns_sections", []) == want["bns_sections"], case["why"]
    assert state.get("route") == want["route"], case["why"]

    got_reasons = {
        item["ipc_section"]: item["reason"] for item in state.get("review_queue", [])
    }
    assert got_reasons == want["review_reasons"], case["why"]
    assert not state.get("errors"), state.get("errors")


# ---------------------------------------------------------------------------
# the mapping safety policy
# ---------------------------------------------------------------------------


def test_auto_apply_requires_high_confidence_and_no_review_flag(ipc_map):
    """The core legal-safety invariant of the whole mapping layer."""
    for row in ipc_map._rows.values():  # noqa: SLF001 -- checking every row on purpose
        if ipc_map.auto_applicable(row):
            assert row.confidence == "high", f"IPC {row.ipc_section} auto-applied at {row.confidence}"
            assert not row.needs_review, f"IPC {row.ipc_section} auto-applied while flagged"
            assert row.is_bns_section, f"IPC {row.ipc_section} auto-applied a non-BNS target"


def test_no_flagged_row_can_ever_be_auto_applied(ipc_map):
    """All 29 needs_review rows must land in the review queue, none in output."""
    flagged = [r.ipc_section for r in ipc_map._rows.values() if r.needs_review]  # noqa: SLF001
    assert len(flagged) == 29, f"expected 29 flagged rows, found {len(flagged)}"

    result = ipc_map.apply(flagged)
    assert result.bns_sections == [], "a flagged row leaked into the auto-applied output"
    assert result.n_review == len(flagged)


def test_sentinel_targets_are_never_emitted(ipc_map):
    """OMITTED / VERIFY / PC Act rows must not become BNS sections."""
    result = ipc_map.apply(["13", "155", "156", "161", "164", "190", "482"])
    assert result.bns_sections == []
    reasons = {i.ipc_section: i.reason for i in result.review_queue}
    assert reasons["13"] is ReviewReason.REPEALED_NO_EQUIVALENT
    assert reasons["161"] is ReviewReason.OTHER_STATUTE
    assert reasons["164"] is ReviewReason.OTHER_STATUTE
    for s in ("155", "156", "190", "482"):
        assert reasons[s] is ReviewReason.MAPPING_UNVERIFIED


def test_unknown_ipc_section_goes_to_review(ipc_map):
    result = ipc_map.apply(["99999"])
    assert result.bns_sections == []
    assert result.review_queue[0].reason is ReviewReason.NOT_IN_TABLE


def test_apply_is_order_preserving_and_deduplicating(ipc_map):
    """Stable output is what makes golden pinning possible."""
    r1 = ipc_map.apply(["302", "379", "302", "379"])
    r2 = ipc_map.apply(["302", "379"])
    assert r1.bns_sections == r2.bns_sections == ["BNS 103(1)", "BNS 303(2)"]


def test_mapping_coverage_matches_documented_counts(ipc_map):
    """data/mapping/README.md claims 100 rows, 71 high-confidence."""
    cov = ipc_map.coverage()
    assert cov["total"] == 100
    assert cov["needs_review"] == 29
    # 71 high-confidence unflagged rows, minus the 2 whose target is not a BNS
    # section at all: IPC 13 -> OMITTED and IPC 161 -> PC Act 1988.
    assert cov["auto_applicable"] == 69
    # 7 non-BNS targets overall: 1 OMITTED + 4 VERIFY + 2 PC Act.
    assert cov["non_bns_targets"] == 7


# ---------------------------------------------------------------------------
# cognizability routing
# ---------------------------------------------------------------------------


def test_cognizable_dominates_mixed_case():
    """A case with both classes is an FIR, not a CSR."""
    result = route_case(["BNS 103(1)", "BNS 351(2)"])
    assert result.route is Route.FIR


def test_all_non_cognizable_is_csr():
    result = route_case(["BNS 115(2)", "BNS 351(2)"])
    assert result.route is Route.CSR


def test_unknown_section_never_silently_becomes_csr():
    """An unclassified section could still be cognizable -- ask the officer."""
    result = route_case(["BNS 999"])
    assert result.route is Route.OFFICER_REVIEW
    assert result.unknown_sections == ["BNS 999"]


def test_empty_sections_route_to_officer():
    assert route_case([]).route is Route.OFFICER_REVIEW


def test_pending_review_items_block_a_csr_decision():
    """Non-cognizable sections plus an unresolved mapping item is not a CSR."""
    assert route_case(["BNS 115(2)"], has_review_items=False).route is Route.CSR
    assert route_case(["BNS 115(2)"], has_review_items=True).route is Route.OFFICER_REVIEW


def test_sub_clause_and_composite_sections_resolve_against_the_schedule():
    assert classify_section("BNS 103(1)") is Cognizability.COGNIZABLE
    assert classify_section("BNS 80") is Cognizability.COGNIZABLE           # derived from 80(2)
    assert classify_section("BNS 324(4)") is Cognizability.NON_COGNIZABLE
    assert classify_section("BNS 324(5)") is Cognizability.COGNIZABLE
    assert classify_section("BNS 324(4),(5)") is Cognizability.UNKNOWN     # parts disagree
    assert classify_section("BNS 324") is Cognizability.UNKNOWN            # sub-clauses differ
    assert classify_section("BNS 3(5)") is Cognizability.UNKNOWN           # not an offence
    assert classify_section("not a section") is Cognizability.UNKNOWN


def test_conditional_entries_resolve_only_with_the_fact_they_need():
    """303(2) theft: the Schedule says non-cognizable under Rs 5,000."""
    assert classify_section("BNS 303(2)") is Cognizability.CONDITIONAL
    assert classify_section("BNS 303(2)", {"property_value_inr": 15000}) is Cognizability.COGNIZABLE
    assert classify_section("BNS 303(2)", {"property_value_inr": 4999.5}) is Cognizability.NON_COGNIZABLE
    assert classify_section("BNS 303(2)", {"something_else": 1}) is Cognizability.CONDITIONAL
    # s.85 cruelty is conditional on *who reports*; no extracted fact resolves it
    assert classify_section("BNS 85", {"property_value_inr": 1}) is Cognizability.CONDITIONAL
    r = route_case(["BNS 303(2)"])
    assert r.route is Route.OFFICER_REVIEW
    assert "5,000" in r.rationale and r.unknown_sections == ["BNS 303(2)"]
    assert route_case(["BNS 303(2)"], context={"property_value_inr": 9000}).route is Route.FIR
    assert route_case(["BNS 303(2)"], context={"property_value_inr": 900}).route is Route.CSR


# ---------------------------------------------------------------------------
# graph plumbing
# ---------------------------------------------------------------------------


def test_audio_input_runs_the_asr_node(graph):
    """Audio entry populates the transcript and still reaches a routing decision."""
    from fir.orchestrator.graph import run_audio

    state = run_audio(graph, "fake.wav")
    assert state["transcript"] == "the accused stole a mobile phone worth Rs 15,000"
    assert state["narrative"] == state["transcript"]
    assert state["bns_sections"] == ["BNS 303(2)"]
    assert state["route"] == "FIR"


def test_text_input_skips_the_asr_node(graph):
    """The conditional entry edge must bypass ASR for written complaints."""
    from fir.orchestrator.graph import run_text

    state = run_text(graph, "the accused stole a mobile phone worth Rs 15,000")
    assert "transcript" not in state
    assert state["bns_sections"] == ["BNS 303(2)"]


def test_label_space_round_trips():
    """Vector positions must agree with the section numbers they decode to."""
    import numpy as np

    from fir.statute.labels import LabelSpace

    space = LabelSpace.from_vocab()
    assert len(space) == 100

    scores = np.zeros(len(space), dtype=np.float32)
    idx = space.sections.index("302")
    scores[idx] = 0.9
    assert space.decode(scores, 0.5) == ["302"]


# ---------------------------------------------------------------------------
# the officer-facing decision record
# ---------------------------------------------------------------------------


def test_graph_state_validates_into_a_decision_record(graph):
    """The typed contract Stage D and Stage E will consume."""
    from fir.orchestrator.graph import run_text, to_decision

    state = run_text(graph, "The victim was harassed for dowry and later found dead.")
    decision = to_decision(state)

    assert decision.route == "FIR"
    assert [s.bns_section for s in decision.suggested_sections] == [
        "BNS 103(1)", "BNS 80", "BNS 85", "BNS 3(5)",
    ]
    # every suggestion keeps the IPC section it came from -- provenance is the
    # point of the record
    assert all(s.ipc_section for s in decision.suggested_sections)
    assert decision.is_auto_resolvable
    # ...and the officer is still the author of record regardless
    assert decision.requires_officer_action


def test_held_back_sections_appear_as_review_flags_not_suggestions(graph):
    from fir.orchestrator.graph import run_text, to_decision

    decision = to_decision(run_text(graph, "A public servant demanded a bribe."))
    assert decision.suggested_sections == []
    assert [f.reason for f in decision.review_flags] == ["maps_to_other_statute"]
    assert decision.route == "OFFICER_REVIEW"
    assert not decision.is_auto_resolvable


# ---------------------------------------------------------------------------
# stage B + D nodes: the graph now runs through to a rendered IF-1
# ---------------------------------------------------------------------------


def test_graph_runs_end_to_end_to_a_rendered_form(graph):
    from fir.orchestrator.graph import run_text

    state = run_text(
        graph,
        "On 8.5.2026 at 10:30 pm the accused stole a mobile phone worth "
        "Rs 15,000 from my shop. My number is 9876543210.",
    )
    assert not state.get("errors"), state.get("errors")

    # stage B floor ran
    kinds = {sp["kind"] for sp in state["extracted_spans"]}
    assert kinds == {"date", "time", "amount_inr", "phone"}

    # stage D produced a record and a form
    rec = state["fir_record"]
    assert rec["schema_version"] == "if1/v1"
    assert rec["status"] == "draft"
    assert rec["occurrence"]["from_datetime"]["value"] == "2026-05-08T22:30"
    assert rec["complainant"]["contact"]["value"] == "9876543210"
    assert rec["total_property_value_inr"]["value"] == 15000.0
    assert [c["section"] for c in rec["acts_sections"]] == ["303(2)"]
    assert rec["cognizability"]["route"] == "FIR"

    form = state["if1_text"]
    assert "FIRST INFORMATION REPORT" in form
    assert "s.303(2)" in form
    assert "8.5.2026 10:30 pm" in form
    assert "13. Action taken : ________" in form   # officer item untouched

    assert set(state["fill_report"]["filled"]) == {
        "occurrence.from_datetime", "complainant.contact", "total_property_value_inr"
    }


def test_graph_form_for_a_review_case_has_no_sections_and_routes_to_officer(graph):
    from fir.orchestrator.graph import run_text

    state = run_text(graph, "A public servant demanded a bribe on 1.1.2026.")
    rec = state["fir_record"]
    assert rec["acts_sections"] == []
    assert rec["cognizability"]["route"] == "officer_review"
    assert "no section could be applied -- officer to determine" in state["if1_text"]
    # but extraction still filled what it could
    assert rec["occurrence"]["from_datetime"]["value"] == "2026-01-01"


# ---------------------------------------------------------------------------
# the BNSS Schedule loader: real CSV replaces the stub when present
# ---------------------------------------------------------------------------


def test_schedule_is_the_gazette_table_and_says_so():
    from fir.statute.cognizability import schedule, schedule_source

    s = schedule()
    assert not s.is_stub
    assert s.n_rows == 438                      # scripts/build_bnss_schedule.py output
    assert "bnss_schedule1.csv" in schedule_source() and "STUB" not in schedule_source()
    assert s.source_refs["103(1)"].startswith("BNSS 2023 First Schedule")
    assert "85" in s.notes and "person aggrieved" in s.notes["85"]


def test_schedule_falls_back_to_the_stub_and_says_so(tmp_path, monkeypatch):
    from fir.statute import cognizability as cog

    monkeypatch.setattr(cog, "DATA_DIR", tmp_path)   # no CSV here
    cog.schedule.cache_clear()
    try:
        assert cog.schedule().is_stub
        assert "STUB" in cog.schedule_source()
        assert "not the official table" in cog.schedule_source()
    finally:
        cog.schedule.cache_clear()


def test_schedule_csv_replaces_stub(tmp_path, monkeypatch):
    """Drop in bnss_schedule1.csv and the stub is ignored, including for a
    section the stub had a different opinion about."""

    from fir.statute import cognizability as cog

    csv_dir = tmp_path / "statutes"
    csv_dir.mkdir()
    (csv_dir / "bnss_schedule1.csv").write_text(
        "bns_section,cognizable,bailable,triable_by,source_ref\n"
        "115,cognizable,Y,Magistrate,Gazette p.412\n"       # stub says NON-cognizable
        "115(2),non_cognizable,Y,Magistrate,Gazette p.412\n"  # clause override
        "999,Y,,,\n"
        "303,,,,\n",                                          # blank -> unknown
        encoding="utf-8",
    )
    monkeypatch.setattr(cog, "DATA_DIR", tmp_path)
    cog.schedule.cache_clear()
    try:
        s = cog.schedule()
        assert not s.is_stub and s.n_rows == 4
        assert cog.classify_section("BNS 115") is cog.Cognizability.COGNIZABLE     # CSV wins
        assert cog.classify_section("BNS 115(2)") is cog.Cognizability.NON_COGNIZABLE  # clause override
        assert cog.classify_section("BNS 115(1)") is cog.Cognizability.COGNIZABLE  # falls to base
        assert cog.classify_section("BNS 999") is cog.Cognizability.COGNIZABLE     # Y accepted
        assert cog.classify_section("BNS 303") is cog.Cognizability.UNKNOWN        # blank never defaults
        assert cog.classify_section("BNS 103") is cog.Cognizability.UNKNOWN        # stub row NOT consulted
        assert "bnss_schedule1.csv" in cog.schedule_source()
        assert s.source_refs["115"] == "Gazette p.412"
    finally:
        cog.schedule.cache_clear()
