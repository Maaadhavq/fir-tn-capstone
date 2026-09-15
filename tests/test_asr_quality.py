"""Transcript quality guards, pinned against real Whisper output.

The fixture strings in tests/fixtures/asr_quality_cases.json are the actual
hypotheses faster-whisper large-v3 produced on FLEURS-ta clips 1916, 1996, 1731
and 1690 in the 60-clip baseline of 2026-08-20. They are kept verbatim so that
the guard is tested against the failure modes we have actually seen, not ones
we imagined.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fir.asr.quality import (  # noqa: E402
    TranscriptFlag,
    assess,
    duplicated_fraction,
    foreign_script_ratio,
)

CASES = json.loads(
    (REPO_ROOT / "tests" / "fixtures" / "asr_quality_cases.json").read_text(encoding="utf-8")
)


# ---------------------------------------------------------------------------
# the four real clips
# ---------------------------------------------------------------------------


def test_repetition_loop_is_flagged():
    """Clip 1916: 45.9s, decoder replayed the first window. 226% WER."""
    rep = assess(CASES["repetition_loop_1916"])
    assert TranscriptFlag.REPETITION_LOOP in rep.flags
    assert rep.duplicated_fraction > 0.30, "replay should duplicate a large share"
    assert TranscriptFlag.FOREIGN_SCRIPT not in rep.flags


def test_foreign_script_drift_is_flagged():
    """Clip 1996: mid-sentence slide into Cyrillic and Greek despite language='ta'."""
    rep = assess(CASES["foreign_script_1996"])
    assert TranscriptFlag.FOREIGN_SCRIPT in rep.flags
    assert "CYRILLIC" in rep.foreign_scripts
    assert rep.foreign_count >= 7
    assert TranscriptFlag.REPETITION_LOOP not in rep.flags


def test_segmentation_artifact_is_not_flagged():
    """Clip 1731: WER 150% but CER 13% -- agglutination boundaries, not a fault.

    This is the case that justifies the guard's existence as something other
    than a WER threshold: WER says this clip is worse than the hallucinated one,
    and it is essentially correct.
    """
    rep = assess(CASES["segmentation_artifact_1731"])
    assert rep.ok, rep.summary()


def test_clean_transcript_is_not_flagged():
    rep = assess(CASES["clean_1690"])
    assert rep.ok, rep.summary()
    assert rep.duplicated_fraction == 0.0
    assert rep.foreign_count == 0


def test_same_sentence_other_speaker_is_clean():
    """FLEURS records each sentence with several speakers. Clip 1996's other
    take decoded cleanly (WER 52%), while this speaker's slid into Cyrillic.
    The guard must separate the two -- it is judging the *decode*, not the text.
    """
    assert assess(CASES["clean_1996_other_speaker"]).ok
    assert not assess(CASES["foreign_script_1996"]).ok


# ---------------------------------------------------------------------------
# the primitives
# ---------------------------------------------------------------------------


def test_empty_transcript_is_flagged():
    assert TranscriptFlag.EMPTY in assess("").flags
    assert TranscriptFlag.EMPTY in assess("   ").flags


def test_code_switched_latin_is_allowed():
    """Real Tamil complaints mix in English. Latin must never trip the guard."""
    text = "அவர் என் mobile phone ஐ திருடினார் police station க்கு வந்தேன்"
    rep = assess(text)
    assert rep.ok, rep.summary()
    ratio, offenders = foreign_script_ratio(text)
    assert ratio == 0.0 and not offenders


def test_a_single_stray_foreign_character_is_tolerated():
    """One stray glyph is noise; the threshold is two."""
    text = "அவர் என் கடையில் இருந்து பணத்தை திருடினார் я"
    assert assess(text).ok


def test_two_foreign_characters_are_enough():
    text = "அவர் என் கடையில் இருந்து பணத்தை திருடினார் яв"
    assert TranscriptFlag.FOREIGN_SCRIPT in assess(text).flags


def test_a_genuinely_repeated_short_phrase_is_tolerated():
    """A complainant saying 'he hit me, he hit me' must not read as a decoder loop."""
    text = (
        "அவர் என்னை அடித்தார் அவர் என்னை அடித்தார் பிறகு அவர் என் "
        "பணத்தை எடுத்துக்கொண்டு கடையை விட்டு ஓடினார் நான் உடனே "
        "காவல் நிலையத்திற்கு வந்து புகார் அளித்தேன் அங்கு இருந்த"
    )
    rep = assess(text)
    assert rep.ok, rep.summary()
    assert 0.0 < rep.duplicated_fraction < 0.15


def test_full_replay_is_caught_even_when_short():
    words = "ஒன்று இரண்டு மூன்று நான்கு ஐந்து ஆறு ஏழு எட்டு".split()
    replayed = " ".join(words + words)
    frac, _, _ = duplicated_fraction(replayed, n=3)
    # 16 words -> 14 trigram positions; the 2 that straddle the seam are
    # unique, so a verbatim replay scores 12/14, not 1.0
    assert frac > 0.8
    assert TranscriptFlag.REPETITION_LOOP in assess(replayed).flags


# ---------------------------------------------------------------------------
# propagation to the officer-facing record
# ---------------------------------------------------------------------------


def test_flagged_transcript_blocks_auto_resolution():
    """The whole point: a flagged transcript is never auto-resolvable, even when
    the statute result on its own would be."""
    from fir.schema.statute_decision import StatuteDecision

    clean = StatuteDecision(route="FIR", asr_flags=[])
    flagged = StatuteDecision(route="FIR", asr_flags=["repetition_loop"])
    assert clean.is_auto_resolvable
    assert not flagged.is_auto_resolvable
    assert flagged.requires_officer_action


def test_graph_propagates_asr_flags(monkeypatch):
    """The ASR node must copy the transcript's flags into graph state."""
    from fir.asr.whisper import Transcript
    from fir.orchestrator.graph import build_slice_graph, run_audio, to_decision
    from fir.statute.ipc_bns_map import IpcBnsMap

    sys.path.insert(0, str(REPO_ROOT))
    from tests.stubs import GOLDEN_RULES, StubStatuteClassifier

    class FlaggingAsr:
        resolved_device = "stub"

        def transcribe(self, path):
            return Transcript(
                text="the accused stole a mobile phone worth Rs 15,000",
                language="ta",
                quality=assess(CASES["foreign_script_1996"]),
            )

    graph = build_slice_graph(
        classifier=StubStatuteClassifier(rules=GOLDEN_RULES),
        asr=FlaggingAsr(),
        ipc_map=IpcBnsMap.load(),
    )
    state = run_audio(graph, "fake.wav")
    assert state["asr_flags"] == ["foreign_script"]
    # statute side still worked ...
    assert state["bns_sections"] == ["BNS 303(2)"]
    assert state["route"] == "FIR"
    # ... but the record refuses to call it resolved
    assert not to_decision(state).is_auto_resolvable


def test_decoder_makes_one_pass_at_temperature_zero_by_default():
    """The library's temperature-fallback ladder re-decodes low-confidence
    windows at higher *sampling* temperature: 5x the wall-clock on Tamil and
    the path fabricated text comes from (finding 6f). Off unless asked for."""
    from fir.asr.whisper import WhisperAsr

    assert WhisperAsr().temperature == 0.0
    assert WhisperAsr(temperature=[0.0, 0.2, 0.4]).temperature == [0.0, 0.2, 0.4]
    assert WhisperAsr().batch_size == 8            # batched VAD-chunk decoding (finding 6f)
    assert WhisperAsr(batch_size=0).batch_size == 0


def test_replay_loop_brakes_are_configurable_and_off_unless_measured():
    from fir.asr.whisper import WhisperAsr

    a = WhisperAsr()
    assert a.repetition_penalty >= 1.0 and a.no_repeat_ngram_size >= 0     # whatever the sweep chose
    b = WhisperAsr(repetition_penalty=1.2, no_repeat_ngram_size=5, model_name="x/y")
    assert (b.repetition_penalty, b.no_repeat_ngram_size, b.model_name) == (1.2, 5, "x/y")
