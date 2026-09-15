"""Element-wise justification (v0, rule-based): yes with provenance, or unclear. Never no."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from fir.statute.elements import (  # noqa: E402
    ELEMENTS,
    check_elements,
    covered_sections,
    summarise,
)

THEFT = "On 8.5.2026 an unknown man stole my mobile phone from my shop without my knowledge."
DOWRY = ("She was married three years ago. Her husband and mother-in-law harassed her for "
         "dowry and she was found dead by hanging.")


def test_theft_all_elements_evidenced_with_provenance():
    checks = check_elements("379", THEFT, utterance_id="u1")
    assert [c.satisfied for c in checks] == ["yes"] * 4
    for c in checks:
        assert c.supporting_provenance, c.element
        for p in c.supporting_provenance:
            assert p.utterance_id == "u1"
            assert THEFT[p.transcript_char_start:p.transcript_char_end]   # offsets are real
    assert "'stole'" in checks[0].justification


def test_missing_ingredient_is_unclear_not_no():
    """The narrative says death but nothing about intent. That is *unclear* --
    the officer must ask -- never *no*, because silence is not evidence."""
    checks = {c.element: c for c in check_elements("302", DOWRY)}
    assert checks["death caused"].satisfied == "yes"
    intent = next(c for k, c in checks.items() if k.startswith("act done with intention"))
    assert intent.satisfied == "unclear"
    assert "establish:" in intent.justification
    assert intent.supporting_provenance == []
    assert not any(c.satisfied == "no" for c in checks.values())


def test_dowry_death_fully_evidenced():
    checks = check_elements("304B", DOWRY)
    assert all(c.satisfied == "yes" for c in checks), [(c.element, c.satisfied) for c in checks]
    assert summarise(checks) == "4/4 elements evidenced"


def test_tamil_cues_work():
    ta = "அவர் என் போன் திருடி எடுத்துச் சென்றார். என்னிடம் அனுமதி இல்லாமல்."
    checks = {c.element: c.satisfied for c in check_elements("379", ta)}
    assert checks["dishonest intention"] == "yes"
    assert checks["moveable property"] == "yes"
    assert checks["without consent"] == "yes"


def test_uncovered_section_returns_empty_and_says_so():
    assert check_elements("120B", THEFT) == []
    assert summarise([]) == "elements not analysed"
    assert "120B" not in covered_sections()


def test_every_element_has_at_least_one_cue_and_compiles():
    for sec, els in ELEMENTS.items():
        for el in els:
            assert el.cues, f"{sec}: {el.name} has no cues"
            assert check_elements(sec, " ".join(el.cues[:1]).replace("\\b", "").replace("\\", "")) or True


def test_no_yes_on_an_unrelated_narrative():
    """A narrative about a phone theft must not evidence dowry-death elements."""
    checks = check_elements("304B", THEFT)
    yes = [c.element for c in checks if c.satisfied == "yes"]
    assert yes == [], yes


def test_elements_reach_the_form_and_record():
    from fir.orchestrator.graph import build_slice_graph, run_text
    from fir.statute.ipc_bns_map import IpcBnsMap
    from tests.stubs import GOLDEN_RULES, StubStatuteClassifier

    g = build_slice_graph(classifier=StubStatuteClassifier(rules=GOLDEN_RULES), ipc_map=IpcBnsMap.load())
    st = run_text(g, DOWRY)
    rec = st["fir_record"]
    by_ipc = {c["ipc_source"]: c for c in rec["acts_sections"]}
    assert len(by_ipc["304B"]["elements"]) == 4
    assert all(e["satisfied"] == "yes" for e in by_ipc["304B"]["elements"])
    assert by_ipc["34"]["elements"] == []              # not covered: honest empty

    form = st["if1_text"]
    assert "[x] death of a woman" in form
    assert "[?] act done with intention" in form
    assert "establish: intention or knowledge" in form
    assert "(elements not analysed for this section)" in form
