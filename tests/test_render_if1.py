"""The IF-1 renderer: fixed form, deterministic, officer items left blank."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from fir.instantiate.render import BLANK, render_if1  # noqa: E402
from fir.schema.if1 import (  # noqa: E402
    Accused,
    ExtractedField,
    FirRecord,
    GroundingEntry,
    Narrative,
    Provenance,
    SectionCandidate,
)


def _sample() -> FirRecord:
    rec = FirRecord(record_id="fir-test000000", created_at="2026-09-13T00:00:00Z")
    rec.jurisdiction.district = ExtractedField.extracted("Chennai", confidence=0.95)
    rec.complainant.name = ExtractedField.extracted(
        "Kumar", ta="குமார்", confidence=0.92,
        provenance=[Provenance(transcript_char_start=0, transcript_char_end=5)],
    )
    rec.complainant.relative_type = "father"
    rec.acts_sections = [
        SectionCandidate(act="BNS_2023", section="303(2)", description_en="Theft",
                         score=0.9, cognizable=True, ipc_source="379"),
        SectionCandidate(act="BNS_2023", section="356(2)", description_en="Defamation",
                         score=0.2, ipc_source="500", mapping_ambiguous=True),
    ]
    rec.accused = [Accused(unknown=True)]
    rec.narrative = Narrative(
        en="An unknown man stole a phone.",
        grounding=[GroundingEntry(sentence_id="s1", supported=True)],
    )
    rec.cognizability.decision = "cognizable"
    rec.cognizability.route = "FIR"
    return rec


def test_render_is_deterministic():
    a, b = render_if1(_sample()), render_if1(_sample())
    assert a == b


def test_all_fifteen_items_present_in_order():
    out = render_if1(_sample())
    positions = [out.index(f"\n{n}. " if n >= 10 else f"\n{n}.  ") for n in range(1, 16)]
    assert positions == sorted(positions), "IF-1 items must appear in numeric order"


def test_officer_items_are_always_blank():
    """Items 13-15 belong to the officer. The system must never fill them."""
    out = render_if1(_sample())
    for item in ("13. Action taken", "14. Signature", "15. Date & time of dispatch"):
        line = next(ln for ln in out.splitlines() if ln.startswith(item))
        assert line.rstrip().endswith(BLANK), line


def test_empty_slots_render_as_visible_blanks_not_omissions():
    out = render_if1(FirRecord())
    assert out.count(BLANK) > 20, "an empty form should be mostly visible blanks"
    assert "6.  Complainant / Informant" in out
    assert "(c) Date / Year of Birth" in out  # empty sub-fields still listed


def test_ambiguous_mapping_is_called_out_on_the_form():
    out = render_if1(_sample())
    assert "MAPPING UNVERIFIED -- officer to confirm" in out
    assert "[from IPC 379]" in out and "[from IPC 500]" in out


def test_plain_mode_has_no_annotations_and_annotated_mode_does():
    plain = render_if1(_sample(), annotate=False)
    annotated = render_if1(_sample(), annotate=True)
    assert "⟨" not in plain
    assert re.search(r"⟨extracted 0\.92 src:1⟩", annotated)
    assert "faithfulness gate: PASS" in annotated


def test_faithfulness_gate_failure_is_shown_when_annotated():
    rec = _sample()
    rec.narrative.grounding[0].supported = False
    out = render_if1(rec, annotate=True)
    assert "faithfulness gate: FAIL -- 0/1 sentences grounded" in out


def test_tamil_surface_text_preferred_when_requested():
    ta = render_if1(_sample(), lang="ta")
    en = render_if1(_sample(), lang="en")
    assert "குமார்" in ta
    # the English rendering falls back to the value when no `en` surface text
    assert "Kumar" in en


def test_csr_advisory_changes_the_title():
    rec = _sample()
    rec.status = "csr_advisory"
    rec.cognizability.route = "CSR_advisory"
    out = render_if1(rec)
    assert "COMMUNITY SERVICE REGISTER -- ADVISORY" in out
    assert "FIRST INFORMATION REPORT" not in out


def test_disclaimer_always_present():
    for rec in (FirRecord(), _sample()):
        out = render_if1(rec)
        assert "Not filed" in out
        assert "author of record" in out
