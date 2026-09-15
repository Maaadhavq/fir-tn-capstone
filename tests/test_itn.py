"""Inverse text normalisation: Tamil and English spoken numbers -> integers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fir.asr.itn import normalise_numbers, number_spans  # noqa: E402


def _one(text):
    sp = number_spans(text)
    assert len(sp) == 1, [(s.text, s.value) for s in sp]
    return sp[0]


# --- Tamil cardinals ---------------------------------------------------------

@pytest.mark.parametrize("text,value", [
    ("ஒன்று", 1), ("இரண்டு", 2), ("மூன்று", 3), ("நான்கு", 4), ("ஐந்து", 5),
    ("ஆறு", 6), ("ஏழு", 7), ("எட்டு", 8), ("ஒன்பது", 9), ("பத்து", 10),
    ("பதினொன்று", 11), ("பன்னிரண்டு", 12), ("பதினைந்து", 15), ("பத்தொன்பது", 19),
    ("இருபது", 20), ("ஐம்பது", 50), ("தொண்ணூறு", 90),
    ("நூறு", 100), ("ஐநூறு", 500), ("தொள்ளாயிரம்", 900),
    ("ஆயிரம்", 1000), ("லட்சம்", 100_000), ("கோடி", 10_000_000),
])
def test_tamil_standalone(text, value):
    assert _one(text).value == value


@pytest.mark.parametrize("text,value", [
    ("இருபத்தைந்து", 25),          # 20 + 5, ஐ -> ை on doubled த
    ("முப்பத்தேழு", 37),           # 30 + 7, ஏ -> ே
    ("எண்பத்தொன்று", 81),          # 80 + 1, ஒ -> ொ
    ("நூற்றைம்பது", 150),          # 100 + 50
    ("நூற்றிருபது", 120),          # 100 + 20, இ -> ி
    ("ஐம்பதாயிரம்", 50_000),       # 50 x 1000, ஆ -> ா on pulli-less த
    ("இரண்டாயிரம்", 2_000),
    ("பத்தாயிரம்", 10_000),
    ("ஓராயிரம்", 1_000),
    ("இருபதாயிரம்", 20_000),
])
def test_tamil_sandhi_compounds(text, value):
    """The joins that a lookup table cannot handle."""
    assert _one(text).value == value


@pytest.mark.parametrize("text,value", [
    ("இரண்டாயிரத்து ஐநூறு", 2_500),
    ("ஒரு லட்சம் இருபதாயிரம்", 120_000),
    ("இரண்டு லட்சம் ஐம்பதாயிரம்", 250_000),
    ("மூன்று கோடி", 30_000_000),
    ("ஐநூற்று ஐம்பது", 550),
])
def test_tamil_multiword_indian_grouping(text, value):
    sp = _one(text)
    assert sp.value == value
    assert sp.text == text            # the whole run is one span


@pytest.mark.parametrize("text,value", [
    ("அஞ்சு", 5), ("நாலு", 4), ("ரெண்டு", 2), ("மூணு", 3), ("ஒன்னு", 1), ("பதினஞ்சு", 15),
])
def test_tamil_colloquial(text, value):
    assert _one(text).value == value


def test_tamil_ordinal_is_flagged():
    sp = _one("பதினைந்தாம் தேதி")
    assert sp.value == 15 and sp.ordinal
    sp = _one("மூன்றாவது")
    assert sp.value == 3 and sp.ordinal


# --- English -----------------------------------------------------------------

@pytest.mark.parametrize("text,value", [
    ("fifty thousand rupees", 50_000),
    ("two lakh fifty thousand", 250_000),
    ("one hundred and twenty five", 125),
    ("three crore", 30_000_000),
    ("twelve", 12),
])
def test_english_cardinals(text, value):
    assert _one(text).value == value


@pytest.mark.parametrize("text,value", [("fifteenth", 15), ("twentieth", 20), ("third", 3), ("eighth", 8)])
def test_english_ordinals(text, value):
    sp = _one(text)
    assert sp.value == value and sp.ordinal


# --- false positives: the important half --------------------------------------

@pytest.mark.parametrize("text", [
    "ஒருவர் வந்தார்",      # ஒருவர் = someone; begins with ஒரு (1)
    "நாள் முழுவதும்",       # நாள் = day; begins with நா (4)
    "முதல் நபர்",           # முதல் = first; begins with மு (3)
    "இருந்து வந்தேன்",      # இருந்து = from; begins with இரு (2)
    "ஐயா சொன்னார்",         # ஐயா = sir; begins with ஐ (5)
    "ஆறுதல் சொன்னார்",      # ஆறுதல் = consolation; begins with ஆறு (6)
    "எண்ணம்",               # எண்ணம் = thought; begins with எண் (8)
    "the accused stole a phone",
    "section 302",
])
def test_no_number_inside_ordinary_words(text):
    assert number_spans(text) == [], [(s.text, s.value) for s in number_spans(text)]


def test_known_ambiguity_aaru_river_vs_six():
    """ஆறு means both 'six' and 'river'. The lexer returns 6; only context can
    disambiguate, and that is the extraction rules' job (require a currency or
    unit word adjacent). Pinned so the limitation is visible, not hidden."""
    assert _one("ஆறு").value == 6


def test_invalid_sequences_are_not_merged():
    """Two tens in a row, or a smaller multiplier before a larger one, are two
    numbers (or noise), never one."""
    sp = number_spans("இருபது முப்பது")             # 20 30 -> two spans
    assert [s.value for s in sp] == [20, 30]
    sp = number_spans("ஆயிரம் லட்சம்")              # thousand lakh -> not one number
    assert [s.value for s in sp] == [1000, 100_000]


# --- offsets and normalisation -------------------------------------------------

def test_offsets_point_at_the_spoken_words():
    text = "அவர் ஐம்பதாயிரம் ரூபாய் எடுத்தார்"
    sp = _one(text)
    assert text[sp.start:sp.end] == "ஐம்பதாயிரம்"


def test_tamil_digit_glyphs_are_not_treated_as_words():
    assert number_spans("௫௦௦ ரூபாய்") == []      # digits are the digit rules' business


def test_normalise_numbers_replaces_in_place():
    out = normalise_numbers("அவர் ஐம்பதாயிரம் ரூபாய் மற்றும் two phones எடுத்தார்")
    assert out == "அவர் 50000 ரூபாய் மற்றும் 2 phones எடுத்தார்"


def test_bare_multiplier_after_a_digit_is_left_for_the_digit():
    """'5 lakh' must stay '5 lakh' -- the scale word belongs to the digit, and
    the amount regex handles digit+scale. Turning it into '5 100000' breaks it."""
    assert number_spans("5 lakh rupees") == []
    assert number_spans("Rs 2 crore") == []
    assert number_spans("5 லட்சம் ரூபாய்") == []
    # but a multiplier after a *word* is still part of a spoken number
    assert _one("five lakh rupees").value == 500_000
    assert _one("ஐந்து லட்சம்").value == 500_000
