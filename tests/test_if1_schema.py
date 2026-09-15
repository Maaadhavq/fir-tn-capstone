"""The IF-1 Pydantic model vs the canonical JSON Schema, and the Stage D bridge.

The Pydantic classes in fir.schema.if1 must not drift from the JSON Schema in
ARCHITECTURE.md §5 (copied verbatim to src/fir/schema/if1_v1.schema.json). The
first tests here walk both and fail on any property present in one but not the
other. The rest exercise the invariants and the decision -> record bridge.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from fir.schema import if1  # noqa: E402
from fir.schema.if1 import (  # noqa: E402
    Accused,
    ExtractedField,
    FirRecord,
    GroundingEntry,
    Narrative,
    Provenance,
)

CANON = json.loads(
    (REPO_ROOT / "src" / "fir" / "schema" / "if1_v1.schema.json").read_text(encoding="utf-8")
)

# canonical $def name -> Pydantic class
_DEF_TO_MODEL = {
    "provenance": if1.Provenance,
    "bilingual": if1.Bilingual,
    "field": if1.ExtractedField,
    "person": if1.Person,
    "accused": if1.Accused,
    "property_item": if1.PropertyItem,
    "section_candidate": if1.SectionCandidate,
}

# canonical top-level object properties that are inline objects -> Pydantic class
_INLINE_TO_MODEL = {
    "jurisdiction": if1.Jurisdiction,
    "occurrence": if1.Occurrence,
    "place_of_occurrence": if1.PlaceOfOccurrence,
    "narrative": if1.Narrative,
    "cognizability": if1.Cognizability,
}


# ---------------------------------------------------------------------------
# schema parity
# ---------------------------------------------------------------------------


def test_top_level_properties_match_canonical_schema():
    canon = set(CANON["properties"])
    model = set(FirRecord.model_fields)
    assert model == canon, (
        f"only in Pydantic: {sorted(model - canon)}; only in JSON Schema: {sorted(canon - model)}"
    )


@pytest.mark.parametrize("def_name,model", list(_DEF_TO_MODEL.items()))
def test_defs_match_canonical_schema(def_name, model):
    canon = set(CANON["$defs"][def_name]["properties"])
    fields = set(model.model_fields)
    assert fields == canon, (
        f"{def_name}: only in Pydantic {sorted(fields - canon)}; "
        f"only in JSON Schema {sorted(canon - fields)}"
    )


@pytest.mark.parametrize("prop,model", list(_INLINE_TO_MODEL.items()))
def test_inline_objects_match_canonical_schema(prop, model):
    canon = set(CANON["properties"][prop]["properties"])
    fields = set(model.model_fields)
    assert fields == canon, (
        f"{prop}: only in Pydantic {sorted(fields - canon)}; "
        f"only in JSON Schema {sorted(canon - fields)}"
    )


def test_canonical_required_fields_have_defaults_or_are_required():
    """Every canonical `required` top-level field must be constructible."""
    rec = FirRecord()
    for name in CANON["required"]:
        assert getattr(rec, name) is not None, name


def test_field_status_enum_matches():
    canon = set(CANON["$defs"]["field"]["properties"]["status"]["enum"])
    from typing import get_args

    assert set(get_args(if1.FieldStatus)) == canon


def test_extra_keys_are_rejected():
    """The form is fixed. A field that is not on the IF-1 must not slip in."""
    with pytest.raises(ValidationError):
        FirRecord(not_a_real_field="x")
    with pytest.raises(ValidationError):
        ExtractedField(value="x", status="extracted", made_up=1)


# ---------------------------------------------------------------------------
# invariants
# ---------------------------------------------------------------------------


def test_fresh_record_is_a_draft_and_never_fileable():
    rec = FirRecord()
    assert rec.status == "draft"
    assert rec.schema_version == "if1/v1"
    assert not rec.is_fileable
    assert rec.record_id.startswith("fir-")


def test_only_the_officer_makes_a_record_fileable():
    rec = FirRecord()
    rec.status = "verified"  # the UI does this; nothing in src/ does
    assert rec.is_fileable


def test_narrative_gate_requires_text_and_full_grounding():
    rec = FirRecord()
    assert not rec.narrative_is_grounded, "empty narrative must fail the gate"

    rec.narrative = Narrative(ta="x", grounding=[])
    assert not rec.narrative_is_grounded, "text with no grounding must fail"

    rec.narrative = Narrative(
        ta="x",
        grounding=[
            GroundingEntry(sentence_id="s1", supported=True),
            GroundingEntry(sentence_id="s2", supported=False),
        ],
    )
    assert not rec.narrative_is_grounded, "one unsupported sentence fails the whole narrative"

    rec.narrative.grounding[1].supported = True
    assert rec.narrative_is_grounded


def test_provenance_offsets_must_be_ordered():
    Provenance(transcript_char_start=0, transcript_char_end=5)
    with pytest.raises(ValidationError):
        Provenance(transcript_char_start=5, transcript_char_end=0)


def test_accused_cannot_be_both_known_and_unknown():
    Accused(known=True)
    Accused(unknown=True)
    with pytest.raises(ValidationError):
        Accused(known=True, unknown=True)


def test_machine_filled_fields_lists_only_extracted_slots():
    rec = FirRecord()
    rec.complainant.name = ExtractedField.extracted("Kumar", ta="குமார்", confidence=0.9)
    rec.occurrence.from_datetime = ExtractedField.extracted("2026-08-20T10:30")
    rec.delay_reason = ExtractedField(value="none", status="officer_added")  # not machine
    paths = rec.machine_filled_fields()
    assert "complainant.name" in paths
    assert "occurrence.from_datetime" in paths
    assert "delay_reason" not in paths
    assert "complainant.address" not in paths  # empty


# ---------------------------------------------------------------------------
# the Stage D bridge: StatuteDecision -> FirRecord
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def graph():
    from fir.orchestrator.graph import build_slice_graph
    from fir.statute.ipc_bns_map import IpcBnsMap
    from tests.stubs import GOLDEN_RULES, StubAsr, StubStatuteClassifier

    return build_slice_graph(
        classifier=StubStatuteClassifier(rules=GOLDEN_RULES),
        asr=StubAsr(default="the accused stole a mobile phone worth Rs 15,000"),
        ipc_map=IpcBnsMap.load(),
    )


def _decide(graph, text):
    from fir.orchestrator.graph import run_text, to_decision

    return to_decision(run_text(graph, text))


def test_bridge_writes_auto_applied_sections_as_bns_candidates(graph):
    from fir.instantiate.from_decision import record_from_decision

    rec = record_from_decision(_decide(graph, "harassed for dowry and found dead"))
    secs = {c.section: c for c in rec.acts_sections}
    assert set(secs) == {"103(1)", "80", "85", "3(5)"}
    for c in secs.values():
        assert c.act == "BNS_2023"
        assert c.ipc_source  # lineage preserved
        assert not c.mapping_ambiguous
    assert secs["103(1)"].ipc_source == "302"
    assert secs["103(1)"].cognizable is True
    assert rec.cognizability.decision in ("cognizable", "mixed")  # 3(5) is 'unknown'
    assert rec.cognizability.route == "FIR"
    assert rec.status == "draft"


def test_bridge_records_held_back_sections_as_ambiguous(graph):
    """IPC 500 -> BNS 356(2) is needs_review. It must appear, flagged, not vanish."""
    from fir.instantiate.from_decision import record_from_decision

    rec = record_from_decision(_decide(graph, "published defamatory statements"))
    assert len(rec.acts_sections) == 1
    c = rec.acts_sections[0]
    assert c.section == "356(2)"
    assert c.ipc_source == "500"
    assert c.mapping_ambiguous is True
    assert c.score <= 0.25
    assert rec.cognizability.route == "officer_review"


def test_bridge_drops_candidates_with_no_bns_target(graph):
    """IPC 161 -> PC Act 1988. There is no BNS section to propose."""
    from fir.instantiate.from_decision import record_from_decision

    rec = record_from_decision(_decide(graph, "public servant demanded a bribe"))
    assert rec.acts_sections == []
    assert rec.cognizability.decision == "undetermined"
    assert rec.cognizability.route == "officer_review"


def test_bridge_marks_csr_route_as_advisory(graph):
    from fir.instantiate.from_decision import record_from_decision

    rec = record_from_decision(_decide(graph, "the neighbour slapped him during an argument"))
    assert rec.cognizability.decision == "non_cognizable"
    assert rec.cognizability.route == "CSR_advisory"
    assert rec.status == "csr_advisory"
    assert not rec.is_fileable


def test_bridge_never_produces_a_verified_record(graph):
    from fir.instantiate.from_decision import apply_decision

    for text in ("harassed for dowry", "stole a phone", "slapped him", "nothing matches"):
        rec = apply_decision(FirRecord(), _decide(graph, text))
        assert rec.status != "verified"
        assert not rec.is_fileable


def test_bridge_ranks_auto_applied_above_ambiguous():
    from fir.instantiate.from_decision import section_candidates
    from fir.schema.statute_decision import (
        ReviewFlag,
        SectionSuggestion,
        StatuteDecision,
    )

    d = StatuteDecision(
        route="FIR",
        suggested_sections=[
            SectionSuggestion(bns_section="BNS 303(2)", ipc_section="379", cognizable="cognizable")
        ],
        review_flags=[
            ReviewFlag(ipc_section="500", reason="flagged_needs_review", suggested_bns="BNS 356(2)")
        ],
        ipc_top_k=[("500", 0.99), ("379", 0.6)],  # model liked 500 more...
    )
    cands = section_candidates(d)
    # ...but the auto-applied one still ranks first: ambiguity caps the score
    assert [c.section for c in cands] == ["303(2)", "356(2)"]
    assert cands[1].score <= 0.25
