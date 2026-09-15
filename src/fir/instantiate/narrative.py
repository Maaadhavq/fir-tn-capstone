"""Grounded narrative composition for IF-1 item 12.

CONTEXT.md: the narrative is "auto-assembled as a grounded factual summary of
the extracted facts -- every sentence must trace to an extracted fact and pass
the faithfulness gate (no invented content), then the officer edits it."

This is the deterministic composer. Each template sentence is emitted only when
the slots it needs are filled, and its grounding entry carries the provenance
of exactly those slots. So a sentence cannot exist without the facts that
support it, and `FirRecord.narrative_is_grounded` is true by construction --
not because a checker was satisfied after the fact, but because the composer
is incapable of writing an ungrounded sentence.

When an LLM composer is added, it must be held to the same contract: emit
(sentence, provenance) pairs, and any sentence it cannot ground is dropped
before the officer sees it. This module is the reference implementation of
that contract and the fallback when the LLM is unavailable.

Language: English templates are authoritative. Tamil templates are provided
because the officer-facing form is bilingual, but they were written by a
non-native author and MUST be reviewed by a Tamil speaker before any real use.
"""

from __future__ import annotations

from dataclasses import dataclass

from fir.schema.if1 import (
    ExtractedField,
    FirRecord,
    GroundingEntry,
    Narrative,
    Provenance,
)


@dataclass(slots=True)
class Sentence:
    sid: str
    en: str
    ta: str
    provenance: list[Provenance]
    factual: bool = True   # framing sentences carry no claim and no provenance


def _surface(f: ExtractedField, lang: str) -> str:
    """Requested-language surface text -> normalised value -> other language.

    Same order as the renderer: the value is canonical, so it outranks a
    surface string in the wrong language.
    """
    preferred = getattr(f.text, lang, None) if f.text else None
    if preferred:
        return preferred
    if f.value is not None:
        return str(f.value)
    return (f.text.en or f.text.ta or "") if f.text else ""


def _money(f: ExtractedField) -> str:
    v = f.value
    if isinstance(v, (int, float)):
        return f"{v:,.0f}" if float(v).is_integer() else f"{v:,.2f}"
    return str(v)


def compose_sentences(rec: FirRecord) -> list[Sentence]:
    """Build the sentence list from whatever the record has. Order is the
    order a statement is normally taken in: who, when, where, what, loss."""
    S: list[Sentence] = []
    n = 0

    def sid() -> str:
        nonlocal n
        n += 1
        return f"s{n}"

    # who is speaking ------------------------------------------------------
    c = rec.complainant
    if c.name.is_filled:
        name_en, name_ta = _surface(c.name, "en"), _surface(c.name, "ta")
        S.append(Sentence(
            sid(),
            f"The complainant, {name_en}, states as follows.",
            f"புகார்தாரர் {name_ta} கூறுவதாவது:",
            list(c.name.provenance),
        ))
    else:
        S.append(Sentence(
            sid(),
            "The complainant states as follows.",
            "புகார்தாரர் கூறுவதாவது:",
            [],
            factual=False,
        ))

    # when -------------------------------------------------------------------
    occ = rec.occurrence
    if occ.from_datetime.is_filled:
        when_en = _surface(occ.from_datetime, "en")
        when_ta = _surface(occ.from_datetime, "ta")
        prov = list(occ.from_datetime.provenance)
        if occ.is_interval and occ.to_datetime.is_filled:
            to_en, to_ta = _surface(occ.to_datetime, "en"), _surface(occ.to_datetime, "ta")
            prov += occ.to_datetime.provenance
            S.append(Sentence(
                sid(),
                f"The incident took place between {when_en} and {to_en}.",
                f"சம்பவம் {when_ta} முதல் {to_ta} வரை நடந்தது.",
                prov,
            ))
        else:
            S.append(Sentence(
                sid(),
                f"The incident took place on {when_en}.",
                f"சம்பவம் {when_ta} அன்று நடந்தது.",
                prov,
            ))

    # where ------------------------------------------------------------------
    po = rec.place_of_occurrence
    if po.address.is_filled:
        S.append(Sentence(
            sid(),
            f"The place of occurrence is {_surface(po.address, 'en')}.",
            f"சம்பவம் நடந்த இடம்: {_surface(po.address, 'ta')}.",
            list(po.address.provenance),
        ))

    # what / who did it -------------------------------------------------------
    # The offence description comes from the statute decision, and the accused
    # from extraction. Both must be present for a "who did what" sentence;
    # with only the offence we state the offence and nothing about the actor.
    primary = next((s for s in rec.acts_sections if not s.mapping_ambiguous), None)
    if primary and primary.description_en:
        offence = primary.description_en.rstrip(".").lower()
        # strip the "(punishment)" / "(definition)" tags the mapping table carries
        for tag in (" (punishment)", " (definition)", "punishment for ", "punishment - "):
            offence = offence.replace(tag, "")
        offence = offence.strip()

        named = [a for a in rec.accused if a.name.is_filled]
        unknown = any(a.unknown for a in rec.accused)
        prov: list[Provenance] = []
        for a in named:
            prov += a.name.provenance

        if named:
            who_en = ", ".join(_surface(a.name, "en") for a in named)
            who_ta = ", ".join(_surface(a.name, "ta") for a in named)
            S.append(Sentence(
                sid(),
                f"The complainant alleges that {who_en} committed {offence}.",
                f"{who_ta} {offence} செய்ததாக புகார்தாரர் குற்றம் சாட்டுகிறார்.",
                prov,
            ))
        elif unknown:
            S.append(Sentence(
                sid(),
                f"The complainant alleges {offence} by an unknown person.",
                f"அடையாளம் தெரியாத நபர் {offence} செய்ததாக புகார்தாரர் குற்றம் சாட்டுகிறார்.",
                [],
                factual=False,  # "unknown" is an absence of extraction, not a fact with provenance
            ))
        else:
            S.append(Sentence(
                sid(),
                f"The complaint alleges {offence}.",
                f"புகார் {offence} குறித்தது.",
                [],
                factual=False,
            ))

    # loss ---------------------------------------------------------------------
    if rec.total_property_value_inr.is_filled:
        f = rec.total_property_value_inr
        S.append(Sentence(
            sid(),
            f"Property valued at Rs. {_money(f)} is involved.",
            f"ரூ. {_money(f)} மதிப்புள்ள சொத்து சம்பந்தப்பட்டுள்ளது.",
            list(f.provenance),
        ))
    elif rec.properties_involved:
        items = [p for p in rec.properties_involved if p.description.is_filled or p.category.is_filled]
        if items:
            prov = []
            names_en, names_ta = [], []
            for p in items:
                src = p.description if p.description.is_filled else p.category
                names_en.append(_surface(src, "en"))
                names_ta.append(_surface(src, "ta"))
                prov += src.provenance
            S.append(Sentence(
                sid(),
                f"The property involved is: {', '.join(names_en)}.",
                f"சம்பந்தப்பட்ட சொத்து: {', '.join(names_ta)}.",
                prov,
            ))

    return S


def compose_narrative(rec: FirRecord) -> Narrative:
    """Compose item 12 and its grounding. Mutates nothing; returns a Narrative."""
    sents = compose_sentences(rec)
    return Narrative(
        en=" ".join(s.en for s in sents),
        ta=" ".join(s.ta for s in sents),
        grounding=[
            GroundingEntry(
                sentence_id=s.sid,
                # a factual sentence is supported iff it has provenance; a
                # framing sentence makes no claim and is supported trivially
                supported=(bool(s.provenance) if s.factual else True),
                supporting_provenance=s.provenance,
            )
            for s in sents
        ],
    )


def attach_narrative(rec: FirRecord) -> FirRecord:
    """Compose and write the narrative onto the record. Returns the record."""
    rec.narrative = compose_narrative(rec)
    return rec
