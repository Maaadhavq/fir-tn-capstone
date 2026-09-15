"""Rule-based extraction of structured slots, with provenance.

Stage B has two learned approaches planned (encoder NER, LLM structured
extraction). This module is the deterministic floor under both: the slots on an
IF-1 that are *patterns* rather than *language* -- dates, times, amounts, phone
numbers, vehicle registrations. Three reasons to do these with rules first:

1. They are where an LLM is most likely to hallucinate plausibly (a date that
   was never said) and where a regex cannot.
2. They are where WER hides the damage. "8.5.2010" mis-heard as "8.5.2016" is
   one substitution in a 300-word transcript and a wrong date on an FIR.
3. Every match has exact character offsets for free, which is what the IF-1
   schema's `provenance` demands of every filled field.

Tamil is handled where the pattern is stable: month names, time-of-day words
(காலை / மதியம் / மாலை / இரவு), the currency word ரூபாய், and the lakh/crore
scale words. Spelled-out numerals with sandhi (ஐம்பதாயிரம் = fifty thousand)
come from fir.asr.itn -- inverse text normalisation, owned by the ASR stage --
and are accepted as amounts only when a currency word is adjacent.

Nothing in this module fills a slot it is not sure of. A date with an ambiguous
day/month order returns both readings and lets the officer choose.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Iterator

from fir.schema.if1 import ExtractedField, Provenance


class Kind(str, Enum):
    DATE = "date"
    TIME = "time"
    AMOUNT_INR = "amount_inr"
    PHONE = "phone"
    VEHICLE = "vehicle"


@dataclass(slots=True)
class Span:
    """One extracted value and where it came from."""

    kind: Kind
    text: str            # verbatim surface form
    start: int           # char offset into the transcript
    end: int
    value: str | float | int | None   # normalised: ISO date, HH:MM, float INR, digits
    confidence: float
    alternatives: list[str] = field(default_factory=list)  # e.g. the other d/m reading

    def provenance(self, utterance_id: str | None = None) -> Provenance:
        return Provenance(
            utterance_id=utterance_id,
            transcript_char_start=self.start,
            transcript_char_end=self.end,
        )

    def to_field(self, utterance_id: str | None = None) -> ExtractedField:
        note = None
        if self.alternatives:
            note = "ambiguous; alternatives: " + ", ".join(self.alternatives)
        f = ExtractedField.extracted(
            self.value,
            ta=self.text if _has_tamil(self.text) else None,
            en=self.text if not _has_tamil(self.text) else None,
            confidence=self.confidence,
            provenance=[self.provenance(utterance_id)],
        )
        f.officer_note = note
        return f


def _has_tamil(s: str) -> bool:
    return any("஀" <= c <= "௿" for c in s)


def _normalised(text: str):
    """Digit-normalised view of `text` (fir.asr.itn) with an offset map back.

    All extractors run their regexes over `.text` so that "பத்து மணி" and
    "10 மணி" hit the same pattern, then translate matches back so `Span`
    offsets -- and therefore provenance -- point at the words actually said.
    """
    from fir.asr.itn import normalise_numbers_mapped

    return normalise_numbers_mapped(text)


def _span(norm, kind: Kind, a: int, b: int, value, conf: float, alts=None) -> Span:
    """Span from a match [a, b) on normalised text: original offsets, original surface."""
    oa, ob = norm.to_original(a, b)
    return Span(kind, norm.original[oa:ob].strip(), oa, ob, value, conf, alts or [])


def _overlaps_spoken(norm, a: int, b: int, ordinal_only: bool = False) -> bool:
    """Does normalised match [a, b) cover any spelled-out number?"""
    oa, ob = norm.to_original(a, b)
    return any(sp.start < ob and oa < sp.end and (sp.ordinal or not ordinal_only)
               for sp in norm.spans)


# ---------------------------------------------------------------------------
# dates
# ---------------------------------------------------------------------------

_MONTHS_EN = {
    m: i for i, m in enumerate(
        ["january", "february", "march", "april", "may", "june", "july",
         "august", "september", "october", "november", "december"], 1)
}
_MONTHS_EN.update({m[:3]: i for m, i in list(_MONTHS_EN.items())})
_MONTHS_EN["sept"] = 9

_MONTHS_TA = {
    "ஜனவரி": 1, "பிப்ரவரி": 2, "மார்ச்": 3, "ஏப்ரல்": 4, "மே": 5, "ஜூன்": 6,
    "ஜூலை": 7, "ஆகஸ்ட்": 8, "செப்டம்பர்": 9, "அக்டோபர்": 10, "நவம்பர்": 11,
    "டிசம்பர்": 12,
}
_MONTH_ALT = "|".join(sorted(map(re.escape, list(_MONTHS_EN) + list(_MONTHS_TA)), key=len, reverse=True))

# 8.5.2010  8/5/2010  8-5-2010  08.05.10
_NUMERIC_DATE = re.compile(r"(?<!\d)(\d{1,2})[./-](\d{1,2})[./-](\d{4}|\d{2})(?!\d)")
# 8 May 2010 · 8th May, 2010 · May 8 2010 · 8 மே 2010 · 2010 மே 8 (year-first is
# natural Tamil order: "2026 மார்ச் 20 அன்று")
_WORD_DATE = re.compile(
    rf"(?<!\d)(?:(\d{{1,2}})(?:st|nd|rd|th)?\s*(?:of\s+)?({_MONTH_ALT})\s*,?\s*(\d{{4}})"
    rf"|({_MONTH_ALT})\s+(\d{{1,2}})(?:st|nd|rd|th)?\s*,?\s*(\d{{4}})"
    rf"|(\d{{4}})\s+({_MONTH_ALT})\s+(\d{{1,2}}))(?!\d)",
    re.IGNORECASE,
)


def _month_num(name: str) -> int | None:
    n = name.lower()
    return _MONTHS_EN.get(n) or _MONTHS_TA.get(name)


def _valid(y: int, m: int, d: int) -> bool:
    try:
        date(y, m, d)
        return True
    except ValueError:
        return False


def _year4(y: str) -> int:
    v = int(y)
    return v if v >= 100 else (2000 + v if v < 50 else 1900 + v)


# Spoken, year-less: "மே எட்டாம் தேதி" / "மே மாதம் 8 தேதி" / "8 தேதி மே" / "8th of May" /
# "May 8". After number normalisation the ordinal is already a digit. The
# negative lookahead refuses a match that a full date pattern already covers.
_YEARLESS_DATE = re.compile(
    rf"(?<!\d)(?:({_MONTH_ALT})\s+(?:மாதம்\s+)?(\d{{1,2}})(?:st|nd|rd|th)?\s*(?:ஆம்\s*)?(?:தேதி)?"
    rf"|(\d{{1,2}})(?:st|nd|rd|th)?\s*(?:ஆம்\s*)?(?:தேதி\s+)?(?:of\s+)?({_MONTH_ALT}))"
    rf"(?!\s*,?\s*\d{{4}})(?![\d\w])",
    re.IGNORECASE,
)


def extract_dates(text: str) -> list[Span]:
    """Dates in numeric, word, or spoken form.

    Numeric dates are read day-first (Indian convention); when the month-first
    reading is also valid it is offered as an alternative rather than silently
    discarded. A month+day with no year is returned as ISO "--MM-DD" at reduced
    confidence: spoken complaints routinely omit the year, and a partial the
    officer can complete beats a blank.
    """
    norm = _normalised(text)
    t = norm.text
    out: list[Span] = []

    for m in _NUMERIC_DATE.finditer(t):
        a, b, y = int(m.group(1)), int(m.group(2)), _year4(m.group(3))
        dmy = _valid(y, b, a)   # day-first
        mdy = _valid(y, a, b)   # month-first
        if not dmy and not mdy:
            continue
        if dmy:
            value = date(y, b, a).isoformat()
            alts = [date(y, a, b).isoformat()] if (mdy and a != b) else []
            conf = 0.9 if not alts else 0.7
        else:
            value = date(y, a, b).isoformat()
            alts, conf = [], 0.6   # only the non-Indian reading was valid
        out.append(_span(norm, Kind.DATE, m.start(), m.end(), value, conf, alts))

    for m in _WORD_DATE.finditer(t):
        if m.group(1):
            d, mon, y = int(m.group(1)), _month_num(m.group(2)), int(m.group(3))
        elif m.group(4):
            mon, d, y = _month_num(m.group(4)), int(m.group(5)), int(m.group(6))
        else:
            y, mon, d = int(m.group(7)), _month_num(m.group(8)), int(m.group(9))
        if mon and _valid(y, mon, d):
            out.append(_span(norm, Kind.DATE, m.start(), m.end(), date(y, mon, d).isoformat(), 0.95))

    covered = [(sp.start, sp.end) for sp in out]
    for m in _YEARLESS_DATE.finditer(t):
        if m.group(1):
            mon, d = _month_num(m.group(1)), int(m.group(2))
        else:
            d, mon = int(m.group(3)), _month_num(m.group(4))
        if not mon or not _valid(2000, mon, d):     # 2000 is a leap year: 29 Feb allowed
            continue
        oa, ob = norm.to_original(m.start(), m.end())
        if any(a < ob and oa < b for a, b in covered):
            continue
        out.append(_span(norm, Kind.DATE, m.start(), m.end(), f"--{mon:02d}-{d:02d}", 0.6,
                         ["year not stated"]))

    return _dedupe(out)


# ---------------------------------------------------------------------------
# times
# ---------------------------------------------------------------------------

_TOD_TA = {"காலை": "am", "மதியம்": "pm", "மாலை": "pm", "இரவு": "pm", "அதிகாலை": "am"}
_TOD_EN = {"morning": "am", "afternoon": "pm", "evening": "pm", "night": "pm", "noon": "pm"}
_TOD_ALT = "|".join(sorted(map(re.escape, list(_TOD_TA) + list(_TOD_EN)), key=len, reverse=True))

# 10:30 · 10.30 · 10:30 pm · 10 pm · 10 மணி · காலை 10 மணி · 10.30 மணிக்கு · around 10 in the morning
_TIME = re.compile(
    rf"(?:({_TOD_ALT})\s+)?(?<!\d)(\d{{1,2}})(?:[:.](\d{{2}}))?"
    rf"(?:\s*(a\.?m\.?|p\.?m\.?|o'?clock|மணிக்கு|மணி(?:யளவில்)?))?"
    rf"(?:\s+(?:in\s+the\s+)?({_TOD_ALT}))?",
    re.IGNORECASE,
)


def extract_times(text: str) -> list[Span]:
    """Clock times, digit or spoken ("பத்து மணி"). A bare number only counts if
    it is anchored by am/pm, மணி, o'clock, or a time-of-day word -- otherwise
    '8' is just a number."""
    norm = _normalised(text)
    out: list[Span] = []
    for m in _TIME.finditer(norm.text):
        tod_before, hh, mm, marker, tod_after = m.groups()
        if not (marker or tod_before or tod_after):
            continue
        h = int(hh)
        if h > 24 or (mm and int(mm) > 59):
            continue
        tod = (tod_before or tod_after or "")
        half = _TOD_TA.get(tod) or _TOD_EN.get(tod.lower()) or None
        mk = (marker or "").lower().replace(".", "")
        if mk.startswith("p"):
            half = "pm"
        elif mk.startswith("a"):
            half = "am"
        if half == "pm" and 1 <= h < 12:
            h += 12
        if half == "am" and h == 12:
            h = 0
        conf = 0.9 if (mm or marker) else 0.75
        if half is None and h <= 12:
            conf -= 0.15   # 12-hour clock with no am/pm resolved: genuinely ambiguous
        out.append(_span(norm, Kind.TIME, m.start(), m.end(),
                         f"{h:02d}:{int(mm or 0):02d}", round(conf, 2)))
    return _dedupe(out)


# ---------------------------------------------------------------------------
# amounts
# ---------------------------------------------------------------------------

_SCALE = {
    "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "லட்சம்": 1e5,
    "crore": 1e7, "crores": 1e7, "கோடி": 1e7,
    "thousand": 1e3, "ஆயிரம்": 1e3, "k": 1e3,
}
_SCALE_ALT = "|".join(sorted(map(re.escape, _SCALE), key=len, reverse=True))
_CUR = r"(?:₹|rs\.?|inr|rupees?|ரூபாய்|ரூ\.?)"
_NUM = r"(\d{1,3}(?:,\d{2,3})+|\d+)(?:\.(\d{1,2}))?"

_AMOUNT = re.compile(
    rf"(?:{_CUR}\s*{_NUM}\s*({_SCALE_ALT})?"          # ₹50,000  Rs 5 lakh
    rf"|{_NUM}\s*({_SCALE_ALT})?\s*{_CUR})",           # 50,000 ரூபாய்  5 லட்சம் ரூபாய்
    re.IGNORECASE,
)


def extract_amounts_inr(text: str) -> list[Span]:
    """Rupee amounts in either currency-first or currency-last order, with
    Indian digit grouping (1,50,000) and lakh/crore scale words. Spoken numbers
    ("ஐம்பதாயிரம் ரூபாய்", "fifty thousand rupees") arrive already normalised to
    digits, so the same regex catches them; a spoken number counts only when a
    currency word is adjacent, since "ஆறு" alone is as likely a river as a six.
    """
    norm = _normalised(text)
    out: list[Span] = []
    for m in _AMOUNT.finditer(norm.text):
        g = m.groups()
        whole, frac, scale = (g[0], g[1], g[2]) if g[0] else (g[3], g[4], g[5])
        if not whole:
            continue
        # a spoken ordinal ("பதினைந்தாம் தேதி") normalises to a digit too; it is a
        # date, not money, even next to ரூபாய்
        if _overlaps_spoken(norm, m.start(), m.end(), ordinal_only=True):
            continue
        n = float(whole.replace(",", "") + (f".{frac}" if frac else ""))
        if scale:
            n *= _SCALE[scale.lower()] if scale.lower() in _SCALE else _SCALE[scale]
        # digits the complainant wrote are surer than a number we parsed from speech
        conf = 0.85 if _overlaps_spoken(norm, m.start(), m.end()) else 0.9
        out.append(_span(norm, Kind.AMOUNT_INR, m.start(), m.end(), round(n, 2), conf))
    return _dedupe(out)


# ---------------------------------------------------------------------------
# phones & vehicles
# ---------------------------------------------------------------------------

# Indian mobile: 10 digits starting 6-9, optional +91 / 0, optional separators
_PHONE = re.compile(r"(?<!\d)(?:\+?91[\s-]?|0)?([6-9]\d{4}[\s-]?\d{5})(?!\d)")
# TN 01 AB 1234 · TN01AB1234 · TN-01-AB-1234 (any state code)
_VEHICLE = re.compile(r"\b([A-Z]{2})[\s-]?(\d{1,2})[\s-]?([A-Z]{1,3})[\s-]?(\d{4})\b")


def extract_phone_numbers(text: str) -> list[Span]:
    out = []
    for m in _PHONE.finditer(text):
        digits = re.sub(r"\D", "", m.group(1))
        out.append(Span(Kind.PHONE, m.group(0).strip(), m.start(), m.end(), digits, 0.9))
    return _dedupe(out)


def extract_vehicle_numbers(text: str) -> list[Span]:
    out = []
    for m in _VEHICLE.finditer(text):
        norm = f"{m.group(1)} {int(m.group(2)):02d} {m.group(3)} {m.group(4)}"
        out.append(Span(Kind.VEHICLE, m.group(0), m.start(), m.end(), norm, 0.85))
    return _dedupe(out)


# ---------------------------------------------------------------------------
# glue
# ---------------------------------------------------------------------------


def _dedupe(spans: list[Span]) -> list[Span]:
    """Drop spans fully contained in a longer span of the same kind."""
    spans = sorted(spans, key=lambda s: (s.start, -(s.end - s.start)))
    out: list[Span] = []
    for s in spans:
        if out and out[-1].kind == s.kind and s.start >= out[-1].start and s.end <= out[-1].end:
            continue
        out.append(s)
    return out


def extract_all(text: str) -> dict[Kind, list[Span]]:
    return {
        Kind.DATE: extract_dates(text),
        Kind.TIME: extract_times(text),
        Kind.AMOUNT_INR: extract_amounts_inr(text),
        Kind.PHONE: extract_phone_numbers(text),
        Kind.VEHICLE: extract_vehicle_numbers(text),
    }


def iter_spans(text: str) -> Iterator[Span]:
    for spans in extract_all(text).values():
        yield from spans
