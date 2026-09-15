"""The grounded narrative composer: no sentence without a fact behind it."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from fir.instantiate.narrative import (  # noqa: E402
    attach_narrative,
    compose_narrative,
    compose_sentences,
)
from fir.schema.if1 import (  # noqa: E402
    Accused,
    ExtractedField,
    FirRecord,
    Provenance,
    SectionCandidate,
)


def _p(a, b):
    return [Provenance(transcript_char_start=a, transcript_char_end=b)]


def _rich() -> FirRecord:
    rec = FirRecord()
    rec.complainant.name = ExtractedField.extracted("Kumar", ta="குமார்", provenance=_p(0, 5))
    rec.occurrence.from_datetime = ExtractedField.extracted(
        "2026-05-08T22:30", en="8.5.2026 10:30 pm", provenance=_p(10, 26)
    )
    rec.place_of_occurrence.address = ExtractedField.extracted("T. Nagar market", provenance=_p(30, 45))
    rec.acts_sections = [
        SectionCandidate(act="BNS_2023", section="303(2)", description_en="Theft (punishment)",
                         score=0.9, ipc_source="379"),
    ]
    rec.accused = [Accused(known=True, name=ExtractedField.extracted("Ravi", provenance=_p(50, 54)))]
    rec.total_property_value_inr = ExtractedField.extracted(15000.0, provenance=_p(60, 67))
    return rec


def test_every_factual_sentence_carries_provenance():
    for s in compose_sentences(_rich()):
        if s.factual:
            assert s.provenance, f"factual sentence without provenance: {s.en}"


def test_rich_record_composes_full_narrative_and_passes_gate():
    rec = attach_narrative(_rich())
    en = rec.narrative.en
    assert "Kumar" in en
    assert "8.5.2026 10:30 pm" in en
    assert "T. Nagar market" in en
    assert "Ravi committed theft" in en
    assert "Rs. 15,000" in en
    assert rec.narrative_is_grounded
    assert len(rec.narrative.grounding) == 5
    assert all(g.supported for g in rec.narrative.grounding)


def test_grounding_provenance_points_back_at_the_right_slots():
    rec = attach_narrative(_rich())
    by_sid = {g.sentence_id: g for g in rec.narrative.grounding}
    # sentence 2 is "when"; its provenance must be the datetime span
    assert by_sid["s2"].supporting_provenance[0].transcript_char_start == 10
    # sentence 5 is the loss; its provenance is the amount span
    assert by_sid["s5"].supporting_provenance[0].transcript_char_start == 60


def test_empty_record_yields_only_the_framing_sentence():
    rec = attach_narrative(FirRecord())
    assert rec.narrative.en == "The complainant states as follows."
    assert len(rec.narrative.grounding) == 1
    # a framing sentence makes no claim, so it is trivially supported ...
    assert rec.narrative.grounding[0].supported
    assert rec.narrative.grounding[0].supporting_provenance == []
    # ... and the record still passes the gate: nothing ungrounded was said
    assert rec.narrative_is_grounded


def test_unfilled_slots_produce_no_sentence():
    rec = FirRecord()
    rec.occurrence.from_datetime = ExtractedField.extracted("2026-05-08", provenance=_p(0, 8))
    n = compose_narrative(rec)
    assert "took place on 2026-05-08" in n.en
    assert "place of occurrence" not in n.en
    assert "Property valued" not in n.en
    assert "alleges" not in n.en


def test_interval_dates_compose_a_between_sentence():
    rec = FirRecord()
    rec.occurrence.from_datetime = ExtractedField.extracted("2026-05-01", provenance=_p(0, 8))
    rec.occurrence.to_datetime = ExtractedField.extracted("2026-05-03", provenance=_p(13, 21))
    rec.occurrence.is_interval = True
    n = compose_narrative(rec)
    assert "between 2026-05-01 and 2026-05-03" in n.en
    g = n.grounding[1]
    assert len(g.supporting_provenance) == 2


def test_unknown_accused_is_stated_without_claiming_provenance():
    rec = FirRecord()
    rec.acts_sections = [SectionCandidate(act="BNS_2023", section="303(2)",
                                          description_en="Theft", score=0.9)]
    rec.accused = [Accused(unknown=True)]
    sents = compose_sentences(rec)
    what = next(s for s in sents if "unknown person" in s.en)
    assert not what.factual          # "unknown" is an absence, not an extracted fact
    assert what.provenance == []
    n = compose_narrative(rec)
    assert all(g.supported for g in n.grounding)


def test_ambiguous_mapping_section_is_not_used_for_the_offence_sentence():
    """A held-back section must not be asserted as the offence in the narrative."""
    rec = FirRecord()
    rec.acts_sections = [
        SectionCandidate(act="BNS_2023", section="356(2)", description_en="Defamation",
                         score=0.2, mapping_ambiguous=True),
    ]
    n = compose_narrative(rec)
    assert "defamation" not in n.en.lower()


def test_punishment_tags_are_stripped_from_offence_wording():
    rec = FirRecord()
    rec.acts_sections = [SectionCandidate(act="BNS_2023", section="103(1)",
                                          description_en="Punishment for murder", score=0.9)]
    rec.accused = [Accused(known=True, name=ExtractedField.extracted("X", provenance=_p(0, 1)))]
    n = compose_narrative(rec)
    assert "X committed murder." in n.en
    assert "punishment" not in n.en.lower()


def test_tamil_narrative_is_emitted_alongside_english():
    rec = attach_narrative(_rich())
    assert rec.narrative.ta
    assert "குமார்" in rec.narrative.ta
    assert len(rec.narrative.ta.split(".")) >= 3


def test_graph_end_to_end_passes_the_faithfulness_gate():
    from fir.orchestrator.graph import build_slice_graph, run_text
    from fir.statute.ipc_bns_map import IpcBnsMap
    from tests.stubs import GOLDEN_RULES, StubStatuteClassifier

    g = build_slice_graph(classifier=StubStatuteClassifier(rules=GOLDEN_RULES),
                          ipc_map=IpcBnsMap.load())
    state = run_text(g, "On 8.5.2026 the accused stole a mobile phone worth Rs 15,000.")
    assert state["narrative_grounded"] is True
    rec = state["fir_record"]
    assert "took place on 8.5.2026" in rec["narrative"]["en"]
    assert "Rs. 15,000" in rec["narrative"]["en"]
    assert "12. F.I.R. Contents" in state["if1_text"]
    assert "took place on 8.5.2026" in state["if1_text"]
