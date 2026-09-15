"""Inverse text normalisation: spoken numbers -> integers, Tamil and English.

ASR emits what was said. A complainant says "ஐம்பதாயிரம் ரூபாய்" (fifty thousand
rupees) and "பதினைந்தாம் தேதி" (the fifteenth); the form needs 50000 and 15.
This module finds spelled-out numbers and returns their value with character
offsets, so the extraction rules can treat "ஐம்பதாயிரம்" and "50,000" alike and
still point the provenance at the words actually spoken.

Tamil numerals compound with sandhi -- the joining changes the stem:

    இருபது   (20) + ஐந்து  (5)   -> இருபத்தைந்து   (25)
    ஐம்பது   (50) + ஆயிரம் (1000) -> ஐம்பதாயிரம்    (50 000)
    நூறு    (100) + ஐம்பது (50)  -> நூற்றைம்பது    (150)
    இரண்டு  (2)  + ஆயிரம் (1000) -> இரண்டாயிரம்    (2 000)

So this is a small grammar, not a lookup table. The parser is:

  1. a greedy left-to-right lexer over a stem table that includes the sandhi
     (joining) forms of each numeral, producing a sequence of (value, kind)
     tokens -- units, tens, hundreds, and multipliers (ஆயிரம், லட்சம், கோடி);
  2. an accumulator with Indian-system grouping (…, thousand, lakh, crore).

Coverage is deliberately the *spoken* register: cardinals up to crores, the
ordinal suffixes -ஆம்/-ஆவது, and colloquial variants that appear in speech
(அஞ்சு for ஐந்து, நாலு for நான்கு, ரெண்டு for இரண்டு). Written Tamil-digit
glyphs (௧௨௩) are converted too. What it does not do: fractions, decimals in
words, or dates like "எட்டாம் தேதி" as full dates -- those belong in the date
rules, which will call this for the numeral.

Every value returned is one the parser is *sure* of. An unparseable run of
number words returns nothing rather than a guess, per the same policy as
fir.extract.rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Kind(str, Enum):
    UNIT = "unit"          # 1-9, 11-19
    TENS = "tens"          # 10, 20 … 90
    HUNDREDS = "hundreds"  # 100 … 900
    MULT = "mult"          # 1 000, 1 00 000, 1 00 00 000


@dataclass(slots=True)
class NumberSpan:
    text: str
    start: int
    end: int
    value: int
    ordinal: bool = False
    lang: str = "ta"


# ---------------------------------------------------------------------------
# Tamil lexicon -- every stem form, longest-first at match time
# ---------------------------------------------------------------------------
#
# Each entry: surface form -> (value, kind). Joining (sandhi) forms are listed
# alongside the standalone form. Where a compound is lexicalised as one word
# in common speech (பதினைந்து, நூற்றைம்பது) it is listed directly.

_TA: dict[str, tuple[int, Kind]] = {}


def _add(value: int, kind: Kind, *forms: str) -> None:
    for f in forms:
        _TA[f] = (value, kind)


# units and their joining forms
_add(1, Kind.UNIT, "ஒன்று", "ஒன்னு", "ஒரு", "ஓர்", "ஓர", "ஒன்ற")
_add(2, Kind.UNIT, "இரண்டு", "ரெண்டு", "இரண்ட", "ரெண்ட", "இரு")
_add(3, Kind.UNIT, "மூன்று", "மூணு", "மூன்ற", "மு")
_add(4, Kind.UNIT, "நான்கு", "நாலு", "நான்க", "நால", "நா")
_add(5, Kind.UNIT, "ஐந்து", "அஞ்சு", "ஐந்த", "அஞ்ச", "ஐ")
_add(6, Kind.UNIT, "ஆறு", "ஆற", "அறு")
_add(7, Kind.UNIT, "ஏழு", "ஏழ", "எழு")
_add(8, Kind.UNIT, "எட்டு", "எட்ட", "எண்")
_add(9, Kind.UNIT, "ஒன்பது", "ஒம்பது", "ஒன்பத", "ஒம்பத")

# 10-19 are lexicalised
_add(10, Kind.TENS, "பத்து", "பத்த")   # பத்த + ாயிரம் = பத்தாயிரம் (10 000)
_add(11, Kind.UNIT, "பதினொன்று", "பதினொன்னு", "பதினொன்ற")
_add(12, Kind.UNIT, "பன்னிரண்டு", "பன்னிரெண்டு", "பன்னிரண்ட")
_add(13, Kind.UNIT, "பதிமூன்று", "பதிமூணு", "பதிமூன்ற")
_add(14, Kind.UNIT, "பதினான்கு", "பதினாலு", "பதினான்க")
_add(15, Kind.UNIT, "பதினைந்து", "பதினஞ்சு", "பதினைந்த")
_add(16, Kind.UNIT, "பதினாறு", "பதினாற")
_add(17, Kind.UNIT, "பதினேழு", "பதினேழ")
_add(18, Kind.UNIT, "பதினெட்டு", "பதினெட்ட")
_add(19, Kind.UNIT, "பத்தொன்பது", "பத்தொம்பது", "பத்தொன்பத")

# tens: standalone, joining-with-pulli (இருபத் + த + ைந்து = 25) and
# joining-without-pulli (இருபத + ாயிரம் = 20 000)
_add(20, Kind.TENS, "இருபது", "இருபத்", "இருபத", "இருவது")
_add(30, Kind.TENS, "முப்பது", "முப்பத்", "முப்பத")
_add(40, Kind.TENS, "நாற்பது", "நாற்பத்", "நாற்பத", "நாப்பது", "நாப்பத்", "நாப்பத")
_add(50, Kind.TENS, "ஐம்பது", "ஐம்பத்", "ஐம்பத", "அம்பது", "அம்பத்", "அம்பத")
_add(60, Kind.TENS, "அறுபது", "அறுபத்", "அறுபத", "அறுவது")
_add(70, Kind.TENS, "எழுபது", "எழுபத்", "எழுபத", "எழுவது")
_add(80, Kind.TENS, "எண்பது", "எண்பத்", "எண்பத")
_add(90, Kind.TENS, "தொண்ணூறு", "தொண்ணூற்ற", "தொண்ணூற்று", "தொண்ணூத்")

# hundreds: lexicalised compounds, standalone and joining (நூறு -> நூற்ற)
_add(100, Kind.HUNDREDS, "நூறு", "நூற்று", "நூற்ற")
_add(200, Kind.HUNDREDS, "இருநூறு", "இருநூற்று", "இருநூற்ற")
_add(300, Kind.HUNDREDS, "முன்னூறு", "முந்நூறு", "முன்னூற்று", "முன்னூற்ற")
_add(400, Kind.HUNDREDS, "நானூறு", "நானூற்று", "நானூற்ற")
_add(500, Kind.HUNDREDS, "ஐநூறு", "ஐந்நூறு", "ஐநூற்று", "ஐநூற்ற", "அஞ்சூறு")
_add(600, Kind.HUNDREDS, "அறுநூறு", "அறுநூற்று", "அறுநூற்ற")
_add(700, Kind.HUNDREDS, "எழுநூறு", "எழுநூற்று", "எழுநூற்ற")
_add(800, Kind.HUNDREDS, "எண்ணூறு", "எண்ணூற்று", "எண்ணூற்ற")
_add(900, Kind.HUNDREDS, "தொள்ளாயிரம்", "தொள்ளாயிரத்")

# multipliers: standalone and joining (ஆயிரம் -> ஆயிரத்)
_add(1_000, Kind.MULT, "ஆயிரம்", "ஆயிரத்", "ஆயிர")
_add(100_000, Kind.MULT, "லட்சம்", "லட்சத்", "லச்சம்", "லச்சத்", "இலட்சம்")
_add(10_000_000, Kind.MULT, "கோடி", "கோடிய")

# --- sandhi: vowel-initial stems take a vowel-sign form when joined ----------
#
# When a numeral beginning with a vowel joins a preceding stem, the initial
# vowel letter becomes the corresponding vowel *sign* on the stem's last
# consonant:  இருபத்து + ஐந்து -> இருபத்தைந்து  (ஐ -> ை on த).
# Rather than hand-list every joined form, derive them from the standalone
# forms. அ has no sign (it is the inherent vowel), so அ-initial words simply
# drop the letter.
_VOWEL_SIGN = {"ஆ": "ா", "இ": "ி", "ஈ": "ீ", "உ": "ு", "ஊ": "ூ",
               "எ": "ெ", "ஏ": "ே", "ஐ": "ை", "ஒ": "ொ", "ஓ": "ோ", "அ": ""}
for _form, _entry in list(_TA.items()):
    if _form and _form[0] in _VOWEL_SIGN:
        _joined = _VOWEL_SIGN[_form[0]] + _form[1:]
        if _joined and _joined not in _TA:
            _TA[_joined] = _entry

# ordinal suffixes (strip and flag)
_ORDINAL_TA = ("ஆவது", "ாவது", "ஆம்", "ாம்", "வது")

# glue that can appear between number words in a compound
_GLUE_TA = ("த்", "த", "ு", "்")

_TA_KEYS = sorted(_TA, key=len, reverse=True)   # after all forms are in

# ---------------------------------------------------------------------------
# English lexicon (code-switched speech)
# ---------------------------------------------------------------------------

_EN: dict[str, tuple[int, Kind]] = {}
for i, w in enumerate(["zero", "one", "two", "three", "four", "five", "six", "seven",
                       "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
                       "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]):
    _EN[w] = (i, Kind.TENS if i == 10 else Kind.UNIT)
for w, v in [("twenty", 20), ("thirty", 30), ("forty", 40), ("fifty", 50), ("sixty", 60),
             ("seventy", 70), ("eighty", 80), ("ninety", 90)]:
    _EN[w] = (v, Kind.TENS)
_EN["hundred"] = (100, Kind.HUNDREDS)
for w, v in [("thousand", 1_000), ("lakh", 100_000), ("lakhs", 100_000), ("lac", 100_000),
             ("lacs", 100_000), ("crore", 10_000_000), ("crores", 10_000_000),
             ("million", 1_000_000)]:
    _EN[w] = (v, Kind.MULT)
_EN_ORDINAL = {"first": 1, "second": 2, "third": 3, "fifth": 5, "eighth": 8, "ninth": 9,
               "twelfth": 12, "twentieth": 20, "thirtieth": 30}



def _en_ordinal(w: str) -> int | None:
    """'fifteenth' -> 15, 'twentieth' -> 20, 'third' -> 3; None if not an ordinal."""
    if w in _EN_ORDINAL:
        return _EN_ORDINAL[w]
    if w.endswith("th"):
        stem = w[:-2]
        if stem in _EN and _EN[stem][1] is not Kind.MULT:
            return _EN[stem][0]
    if w.endswith("ieth"):                      # twentieth -> twenty
        stem = w[:-4] + "y"
        if stem in _EN:
            return _EN[stem][0]
    return None


# Tamil digit glyphs
_TA_DIGITS = str.maketrans("௦௧௨௩௪௫௬௭௮௯", "0123456789")


# ---------------------------------------------------------------------------
# accumulator (Indian grouping)
# ---------------------------------------------------------------------------


def _accumulate(tokens: list[tuple[int, Kind]]) -> int | None:
    """Fold a token sequence into one integer.

    `current` collects units/tens/hundreds; a multiplier scales `current` (or 1
    if nothing precedes it: "ஆயிரம்" alone is 1000) and adds it to `total`.
    Indian grouping means multipliers must be strictly decreasing left to right
    (crore > lakh > thousand); a violation means the words were not one number.
    """
    total = 0
    current = 0
    last_mult = None
    saw_hundred = False
    for value, kind in tokens:
        if kind is Kind.MULT:
            if last_mult is not None and value >= last_mult:
                return None
            total += (current or 1) * value
            current = 0
            saw_hundred = False
            last_mult = value
        elif kind is Kind.HUNDREDS:
            if saw_hundred:
                return None
            # English "two hundred" = current(2) * 100; Tamil hundreds are lexicalised
            current = (current * 100) if (value == 100 and current) else current + value
            saw_hundred = True
        elif kind is Kind.TENS:
            if current % 100 >= 10:       # two tens in a row: not one number
                return None
            current += value
        else:  # UNIT
            if 0 < current % 10 or (current % 100 >= 11 and current % 100 <= 19):
                return None
            current += value
    return total + current


# ---------------------------------------------------------------------------
# Tamil lexer
# ---------------------------------------------------------------------------


def _lex_ta(word: str) -> tuple[list[tuple[int, Kind]], bool] | None:
    """Split one Tamil word into numeral tokens. Returns (tokens, is_ordinal)
    or None if the word is not entirely a numeral."""
    ordinal = False
    for suf in _ORDINAL_TA:
        if word.endswith(suf) and len(word) > len(suf):
            word = word[: -len(suf)]
            ordinal = True
            break
    tokens: list[tuple[int, Kind]] = []
    i = 0
    n = len(word)
    while i < n:
        matched = False
        for key in _TA_KEYS:
            if word.startswith(key, i):
                tokens.append(_TA[key])
                i += len(key)
                matched = True
                break
        if matched:
            continue
        # allow sandhi glue between stems, only if something precedes it
        glued = False
        if tokens:
            for g in _GLUE_TA:
                if word.startswith(g, i):
                    i += len(g)
                    glued = True
                    break
        if not glued:
            return None
    return (tokens, ordinal) if tokens else None


_WORD_RE = re.compile(r"[஀-௿A-Za-z]+")


def number_spans(text: str) -> list[NumberSpan]:
    """All spelled-out numbers in `text`, with offsets and integer values.

    Adjacent number words are merged into one span ("fifty thousand",
    "இரண்டு லட்சம்") when they accumulate to a single valid number.
    """
    text_t = text.translate(_TA_DIGITS)   # Tamil digit glyphs -> ASCII
    spans: list[NumberSpan] = []

    # collect per-word tokens with positions
    words: list[tuple[int, int, list[tuple[int, Kind]], bool, str]] = []
    for m in _WORD_RE.finditer(text_t):
        w = m.group(0)
        low = w.lower()
        if low in _EN:
            words.append((m.start(), m.end(), [_EN[low]], False, "en"))
        elif _en_ordinal(low) is not None:
            v = _en_ordinal(low)
            words.append((m.start(), m.end(), [(v, Kind.TENS if v % 10 == 0 else Kind.UNIT)], True, "en"))
        elif low in ("and",):
            words.append((m.start(), m.end(), [], False, "glue"))
        else:
            lexed = _lex_ta(w)
            if lexed:
                words.append((m.start(), m.end(), lexed[0], lexed[1], "ta"))
            else:
                words.append((m.start(), m.end(), [], False, "other"))

    # merge runs of adjacent number words into one span
    i = 0
    while i < len(words):
        s, e, toks, ordn, lang = words[i]
        if not toks:
            i += 1
            continue
        run_tokens = list(toks)
        run_end = e
        run_ord = ordn
        run_lang = lang
        j = i + 1
        while j < len(words):
            s2, e2, toks2, ordn2, lang2 = words[j]
            if lang2 == "glue":
                j += 1
                continue
            if not toks2 or lang2 != run_lang:
                break
            trial = run_tokens + toks2
            if _accumulate(trial) is None:
                break
            run_tokens = trial
            run_end = e2
            run_ord = ordn2
            j += 1
        value = _accumulate(run_tokens)
        # A run of *only* multipliers directly after a digit ("5 lakh", "2 கோடி")
        # is a scale suffix on that digit, not a number of its own -- the
        # digit-aware amount regex owns it. Left alone it would become "5 100000".
        only_mult = all(k is Kind.MULT for _, k in run_tokens)
        prev = text_t[:s].rstrip()
        if only_mult and prev and prev[-1].isdigit():
            value = None
        if value is not None:
            spans.append(NumberSpan(text[s:run_end], s, run_end, value, run_ord, run_lang))
        i = j if j > i + 1 else i + 1
    return spans


@dataclass(slots=True)
class NormalisedText:
    """Digit-normalised text plus the map back to original offsets.

    Regex extractors run over `.text`; a match at [a, b) there is translated to
    the original span with `.to_original(a, b)`, so provenance always covers the
    words the complainant actually said, not the digits we substituted.
    """

    text: str
    _start: list[int]    # new index -> original index where that char begins
    _end: list[int]      # new index -> original index where that char ends (exclusive)
    spans: list[NumberSpan]
    original: str = ""

    def to_original(self, a: int, b: int) -> tuple[int, int]:
        if a >= b:
            return (self._start[a] if a < len(self._start) else len(self._start), ) * 2
        return self._start[a], self._end[b - 1]


def normalise_numbers_mapped(text: str) -> NormalisedText:
    """Replace spelled-out numbers with digits and keep an offset map."""
    spans = number_spans(text)
    out: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    last = 0
    for sp in spans:
        for i in range(last, sp.start):          # unchanged run
            out.append(text[i])
            starts.append(i)
            ends.append(i + 1)
        digits = str(sp.value) + ("th" if sp.ordinal and sp.lang == "en" else "")
        for _ in digits:                          # every substituted char maps to the whole span
            starts.append(sp.start)
            ends.append(sp.end)
        out.append(digits)
        last = sp.end
    for i in range(last, len(text)):
        out.append(text[i])
        starts.append(i)
        ends.append(i + 1)
    return NormalisedText("".join(out), starts, ends, spans, text)


def normalise_numbers(text: str) -> str:
    """Replace spelled-out numbers with digits, preserving everything else."""
    return normalise_numbers_mapped(text).text
