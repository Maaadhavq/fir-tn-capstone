"""The element cue scan as a recall safety net for the classifier.

Motivated by a real failure (PROGRESS.md finding 5c): the interim translator
rendered "வரதட்சணை" (dowry) as "relief", the classifier predicted only 506, and a
dowry-cruelty complaint (498A, cognizable) routed to CSR. The cue scan reads the
ORIGINAL narrative with bilingual cues and raises what the classifier missed.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from fir.statute.elements import scan_all  # noqa: E402
from tests.stubs import StubStatuteClassifier, StubTranslator  # noqa: E402

DOWRY_TA = ("நான் ஐந்து வருடங்களுக்கு முன்பு திருமணம் செய்தேன். என் கணவர் மற்றும் அவரது தாயார் "
            "வரதட்சணை கேட்டு என்னை தொடர்ந்து கொடுமைப்படுத்தினார்கள். அவர்கள் என்னை அடித்து "
            "கொலை செய்வேன் என்று மிரட்டினார்கள்.")
# what the interim translator actually produced -- "dowry" is gone
DOWRY_EN_BAD = ("I married five years ago. My husband and his mother asked for a relief and "
                "continued beating me. They beat me and threatened to kill me.")


def _graph(rules, translator=None):
    from fir.orchestrator.graph import build_slice_graph
    from fir.statute.ipc_bns_map import IpcBnsMap

    return build_slice_graph(classifier=StubStatuteClassifier(rules=rules),
                             ipc_map=IpcBnsMap.load(), translator=translator)


def test_scan_finds_dowry_cruelty_in_tamil():
    hits = {h.ipc_section: h for h in scan_all(DOWRY_TA)}
    assert "498A" in hits
    assert hits["498A"].fraction == 1.0
    assert any("வரதட்சணை" in c for c in hits["498A"].cues)


def test_threat_to_kill_is_not_murder():
    """கொலை in a threat must not satisfy 302's 'death caused'."""
    assert "302" not in {h.ipc_section for h in scan_all(DOWRY_TA)}
    assert "302" in {h.ipc_section for h in scan_all("She was found dead. He murdered her with a knife.")}


def test_mistranslated_dowry_complaint_is_not_settled_as_csr():
    """The real failure, reproduced: classifier sees only the threat (506,
    non-cognizable) because the translation lost 'dowry'. Without the cue scan
    this is a CSR. With it, the officer is asked about 498A."""
    from fir.orchestrator.graph import run_text, to_decision

    g = _graph({"threat": ["506"]}, StubTranslator(table={DOWRY_TA: DOWRY_EN_BAD}))
    st = run_text(g, DOWRY_TA)
    assert st["predicted_ipc"] == ["506"]
    assert st["bns_sections"] == ["BNS 351(2)"]
    assert st["review_queue"] == []                     # nothing wrong with the mapping
    cues = {h["ipc_section"]: h for h in st["cue_hits"]}
    # BNS 85 is conditional in the Schedule (cognizable when reported by the
    # aggrieved woman or her relative) -- not settled, so it must still raise
    assert cues["498A"]["cognizable"] == "conditional"
    assert st["route"] == "OFFICER_REVIEW"
    assert "IPC 498A" in st["rationale"]
    d = to_decision(st)
    assert not d.is_auto_resolvable
    assert [h.ipc_section for h in d.cue_hits if h.cognizable != "non_cognizable"] == ["498A"]


def test_cue_hits_do_not_disturb_a_correct_fir_route():
    """Plain theft trips 380's cues (a shop is a building). That is a note for
    the officer, not a reason to change an already-cognizable FIR route or to
    add review items."""
    from fir.orchestrator.graph import run_text

    st = run_text(_graph({"stole": ["379"]}), "the accused stole a phone worth Rs 12,000 from my shop")
    assert st["route"] == "FIR"
    assert st["review_queue"] == []
    assert "380" in {h["ipc_section"] for h in st["cue_hits"]}
    assert "the classifier did not predict" not in st["rationale"]


def test_non_cognizable_cue_hit_does_not_force_review():
    """A missed 323 (non-cognizable) on top of an applied 506 (also
    non-cognizable) stays CSR: the cue cannot raise the route."""
    from fir.orchestrator.graph import run_text

    st = run_text(_graph({"threat": ["506"]}), "he hit me and threatened me")
    assert st["route"] == "CSR"
    assert {h["ipc_section"] for h in st["cue_hits"]} == {"323"}


def test_predicted_sections_are_not_reported_as_cue_hits():
    from fir.orchestrator.graph import run_text

    st = run_text(_graph({"dowry": ["498A", "304B"]}),
                  "married three years ago, harassed for dowry by her husband, found dead by hanging")
    assert not {"498A", "304B"} & {h["ipc_section"] for h in st["cue_hits"]}
