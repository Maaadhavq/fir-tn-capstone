"""Write rule-extracted spans into the IF-1 record -- conservatively.

The policy is: fill a slot only when the transcript supports exactly one
reading. Two dates could be an interval or two different events; two phone
numbers could be the complainant's and the accused's. Guessing puts a plausible
wrong value on the form, which is worse than a blank: a blank prompts the
officer to ask, a wrong value invites them to trust it.

So: one match -> fill with provenance. Several matches -> leave the slot empty
and record the candidates in `officer_note` so the verification screen can
offer them as choices. Zero -> untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fir.extract.rules import Kind, Span, extract_all
from fir.schema.if1 import ExtractedField, FirRecord


@dataclass(slots=True)
class FillReport:
    """What was filled, what was left for the officer, and why."""

    filled: dict[str, Span] = field(default_factory=dict)          # slot path -> span
    ambiguous: dict[str, list[Span]] = field(default_factory=dict)  # slot path -> candidates
    unplaced: list[Span] = field(default_factory=list)              # found, no slot for it

    def summary(self) -> str:
        parts = [f"{len(self.filled)} filled"]
        if self.ambiguous:
            parts.append(f"{len(self.ambiguous)} ambiguous")
        if self.unplaced:
            parts.append(f"{len(self.unplaced)} unplaced")
        return ", ".join(parts)


def _date_note(d: Span) -> str | None:
    """Officer note for a date span, matching the kind of ambiguity it carries."""
    if not d.alternatives:
        return None
    if "year not stated" in d.alternatives:
        return "year not stated -- confirm with the complainant"
    return "day/month order ambiguous; alternative: " + ", ".join(d.alternatives)


def _note(slot: ExtractedField, spans: list[Span], what: str) -> None:
    opts = "; ".join(f"'{s.text}' -> {s.value}" for s in spans)
    slot.status = "empty"
    slot.value = None
    slot.officer_note = f"{len(spans)} {what} found, officer to choose: {opts}"


def fill_structured_slots(
    rec: FirRecord, transcript: str, utterance_id: str | None = None
) -> FillReport:
    """Populate occurrence time, complainant contact and property value from
    the transcript. Returns what happened; mutates `rec` in place."""
    found = extract_all(transcript)
    rep = FillReport()

    # --- occurrence: date (+ time) -----------------------------------------
    dates, times = found[Kind.DATE], found[Kind.TIME]
    if len(dates) == 1:
        d = dates[0]
        value = d.value
        prov = [d.provenance(utterance_id)]
        conf = d.confidence
        surface = d.text
        if len(times) == 1:
            t = times[0]
            value = f"{d.value}T{t.value}"
            prov.append(t.provenance(utterance_id))
            conf = min(d.confidence, t.confidence)
            surface = f"{d.text} {t.text}"
        f = ExtractedField.extracted(value, en=surface, confidence=conf, provenance=prov)
        f.officer_note = _date_note(d)
        rec.occurrence.from_datetime = f
        rep.filled["occurrence.from_datetime"] = d
        if len(times) > 1:
            rep.ambiguous["occurrence.from_datetime.time"] = times
    elif (
        len(dates) == 2
        and all(d.value and not str(d.value).startswith("--") for d in dates)  # both have a year
        and dates[0].value <= dates[1].value
    ):
        # two full dates in order: read as an interval, which the IF-1 supports.
        # Year-less dates are excluded: "--05-08" vs "2026-05-10" cannot be ordered.
        a, b = dates
        rec.occurrence.from_datetime = a.to_field(utterance_id)
        rec.occurrence.to_datetime = b.to_field(utterance_id)
        rec.occurrence.is_interval = True
        rep.filled["occurrence.from_datetime"] = a
        rep.filled["occurrence.to_datetime"] = b
    elif len(dates) > 1:
        _note(rec.occurrence.from_datetime, dates, "dates")
        rep.ambiguous["occurrence.from_datetime"] = dates
    elif len(times) == 1 and not dates:
        # a time with no date is still worth recording as a partial
        t = times[0]
        rec.occurrence.from_datetime = ExtractedField.extracted(
            t.value, en=t.text, confidence=t.confidence * 0.8,
            provenance=[t.provenance(utterance_id)],
        )
        rec.occurrence.from_datetime.officer_note = "time only; date not stated"
        rep.filled["occurrence.from_datetime"] = t

    # --- complainant contact ------------------------------------------------
    phones = found[Kind.PHONE]
    if len(phones) == 1:
        rec.complainant.contact = phones[0].to_field(utterance_id)
        rep.filled["complainant.contact"] = phones[0]
    elif len(phones) > 1:
        _note(rec.complainant.contact, phones, "phone numbers")
        rep.ambiguous["complainant.contact"] = phones

    # --- property value -----------------------------------------------------
    amounts = found[Kind.AMOUNT_INR]
    if len(amounts) == 1:
        rec.total_property_value_inr = amounts[0].to_field(utterance_id)
        rep.filled["total_property_value_inr"] = amounts[0]
    elif len(amounts) > 1:
        _note(rec.total_property_value_inr, amounts, "amounts")
        rep.ambiguous["total_property_value_inr"] = amounts

    # --- vehicles: no fixed slot on the IF-1; surface for the narrative stage
    rep.unplaced.extend(found[Kind.VEHICLE])

    return rep
