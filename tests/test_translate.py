"""The Tamil -> English translation node: classifier-only, provenance-preserving."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from fir.translate.indictrans import tamil_share  # noqa: E402
from tests.stubs import (  # noqa: E402
    GOLDEN_RULES,
    StubStatuteClassifier,
    StubTranslator,
)

TA = "மே மாதம் எட்டாம் தேதி ஒருவர் என் போன் திருடிச் சென்றார். ஐம்பதாயிரம் ரூபாய் மதிப்பு."
EN = "On 8 May someone committed theft of my phone worth fifty thousand rupees."


def _graph(translator):
    from fir.orchestrator.graph import build_slice_graph
    from fir.statute.ipc_bns_map import IpcBnsMap

    return build_slice_graph(classifier=StubStatuteClassifier(rules=GOLDEN_RULES),
                             ipc_map=IpcBnsMap.load(), translator=translator)


def test_tamil_share():
    assert tamil_share("என் போன்") == 1.0
    assert tamil_share("my phone") == 0.0
    assert 0.3 < tamil_share("என் mobile phone திருடினார்") < 0.7
    assert tamil_share("12345 ...") == 0.0


def test_tamil_input_is_translated_for_the_classifier_only():
    from fir.orchestrator.graph import run_text

    st = run_text(_graph(StubTranslator(table={TA: EN})), TA)
    assert st["narrative"] == TA                     # original untouched
    assert st["narrative_en"] == EN                  # classifier's view
    assert st["translation"]["model"] == "stub-indictrans"
    assert st["predicted_ipc"] == ["379"]            # classified on the English
    assert st["bns_sections"] == ["BNS 303(2)"]
    # extraction still ran on the *original*: provenance offsets index into Tamil
    amt = st["fir_record"]["total_property_value_inr"]
    assert amt["value"] == 50000.0
    p = amt["provenance"][0]
    assert "ஐம்பதாயிரம்" in TA[p["transcript_char_start"]:p["transcript_char_end"]]


def test_english_input_is_not_translated():
    from fir.orchestrator.graph import run_text

    st = run_text(_graph(StubTranslator()), "the accused stole a mobile phone worth Rs 15,000")
    assert st["translation"] == {}
    assert st["narrative_en"] == st["narrative"]
    assert not st.get("errors")


def test_missing_model_is_reported_not_fatal():
    """Tamil input with no model on disk: the graph still completes, the error
    says how to fix it, and the officer gets the slots with no sections."""
    from fir.orchestrator.graph import run_text

    st = run_text(_graph(StubTranslator(downloaded=False)), TA)
    assert st["narrative_en"] == TA                  # untranslated
    assert any("IndicTrans2 is not downloaded" in e for e in st["errors"])
    assert st["route"] == "OFFICER_REVIEW"
    assert st["fir_record"]["total_property_value_inr"]["value"] == 50000.0   # extraction still ran


def test_no_translator_configured_is_a_clean_noop():
    from fir.orchestrator.graph import run_text

    st = run_text(_graph(None), TA)
    assert st["narrative_en"] == TA and st["translation"] == {} and not st.get("errors")


def test_translated_view_appears_in_format_result():
    from fir.orchestrator.graph import format_result, run_text

    out = format_result(run_text(_graph(StubTranslator(table={TA: EN})), TA))
    assert "translated   : ta->en via stub-indictrans" in out
    assert "narrative_en : On 8 May someone committed theft" in out


def test_retrieval_rank_uses_the_english_view():
    """Statute texts are English; corroboration must rank the translation, not the Tamil."""
    from fir.orchestrator.graph import run_text, to_decision
    from fir.statute.statute_text import StatuteIndex

    d = to_decision(run_text(_graph(StubTranslator(table={TA: EN})), TA))
    s = d.suggested_sections[0]
    assert s.ipc_section == "379"
    assert s.retrieval_rank == 1                         # "committed theft" -> 379 by its own words
    assert StatuteIndex.load().rank_of(TA, "379") is None   # the Tamil alone would have said nothing
