"""The canonical TN FIR (CCTNS IF-1) record — Pydantic form of ARCHITECTURE.md §5.

This is the object Stage B fills, Stage C annotates, Stage D renders, and Stage
E lets the officer edit. Three design rules carried over from the schema:

1. **Every extracted value is wrapped.** An `ExtractedField` carries the value,
   bilingual surface text, a confidence, provenance back to the transcript, and
   an edit status. A bare string in a structured slot would lose the audit trail
   that makes the officer's verification meaningful.

2. **Structured fields are machine-filled; `narrative` is the only composed
   text**, and every sentence of it must point back at provenance. Nothing
   else in the form is generated.

3. **The form is fixed.** These are the IF-1's fields, in the IF-1's order. We
   do not design the document; we fill it.

`tests/test_if1_schema.py` checks that this model's field names cover every
property in the canonical JSON Schema, so the two cannot silently diverge.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# $defs
# ---------------------------------------------------------------------------

FieldStatus = Literal["extracted", "elicited", "officer_edited", "officer_added", "empty"]


class Provenance(BaseModel):
    """Where in the source a value came from. Character offsets are mandatory;
    audio offsets are present only when the source was speech."""

    model_config = ConfigDict(extra="forbid")

    utterance_id: str | None = None
    transcript_char_start: int = Field(ge=0)
    transcript_char_end: int = Field(ge=0)
    audio_start_ms: int | None = Field(default=None, ge=0)
    audio_end_ms: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> "Provenance":
        if self.transcript_char_end < self.transcript_char_start:
            raise ValueError("transcript_char_end precedes transcript_char_start")
        return self


class Bilingual(BaseModel):
    """Tamil verbatim where sourced from speech, plus an English rendering."""

    model_config = ConfigDict(extra="forbid")

    ta: str | None = None
    en: str | None = None


class ExtractedField(BaseModel):
    """Generic wrapper for every machine-filled slot on the form."""

    model_config = ConfigDict(extra="forbid")

    value: str | float | int | bool | None = None
    text: Bilingual | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    provenance: list[Provenance] = Field(default_factory=list)
    status: FieldStatus = "empty"
    officer_note: str | None = None

    @classmethod
    def empty(cls) -> "ExtractedField":
        return cls(value=None, status="empty")

    @classmethod
    def extracted(
        cls,
        value: Any,
        *,
        ta: str | None = None,
        en: str | None = None,
        confidence: float | None = None,
        provenance: list[Provenance] | None = None,
    ) -> "ExtractedField":
        return cls(
            value=value,
            text=Bilingual(ta=ta, en=en) if (ta or en) else None,
            confidence=confidence,
            provenance=provenance or [],
            status="extracted",
        )

    @property
    def is_filled(self) -> bool:
        return self.status != "empty" and self.value is not None

    @property
    def is_machine_filled(self) -> bool:
        return self.status in ("extracted", "elicited")

    @property
    def has_provenance(self) -> bool:
        return bool(self.provenance)


def _empty() -> ExtractedField:
    return ExtractedField.empty()


class Passport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    number: ExtractedField = Field(default_factory=_empty)
    date_place_of_issue: ExtractedField = Field(default_factory=_empty)


RelativeType = Literal["father", "husband", "mother", "spouse", "guardian"]


class Person(BaseModel):
    """Complainant or witness, per the IF-1's particulars block."""

    model_config = ConfigDict(extra="forbid")

    name: ExtractedField = Field(default_factory=_empty)
    relative_type: RelativeType | None = None
    relative_name: ExtractedField = Field(default_factory=_empty)
    dob_or_age: ExtractedField = Field(default_factory=_empty)
    nationality: ExtractedField = Field(default_factory=_empty)
    occupation: ExtractedField = Field(default_factory=_empty)
    contact: ExtractedField = Field(default_factory=_empty)
    address: ExtractedField = Field(default_factory=_empty)
    passport: Passport = Field(default_factory=Passport)


class Accused(BaseModel):
    model_config = ConfigDict(extra="forbid")

    known: bool = False
    unknown: bool = False
    name: ExtractedField = Field(default_factory=_empty)
    alias: ExtractedField = Field(default_factory=_empty)
    relative_name: ExtractedField = Field(default_factory=_empty)
    physical_description: ExtractedField = Field(default_factory=_empty)
    address: ExtractedField = Field(default_factory=_empty)

    @model_validator(mode="after")
    def _known_xor_unknown(self) -> "Accused":
        if self.known and self.unknown:
            raise ValueError("accused cannot be both known and unknown")
        return self


class PropertyItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: ExtractedField = Field(default_factory=_empty)
    description: ExtractedField = Field(default_factory=_empty)
    estimated_value_inr: ExtractedField = Field(default_factory=_empty)


Act = Literal["BNS_2023", "BNSS_2023", "MV_Act", "IT_Act_2000", "TN_PHW_Act", "OTHER"]


class ElementCheck(BaseModel):
    """One legal element of an offence and whether the facts satisfy it.

    This is the research core's output shape: not just "section 303 applies"
    but "dishonest intention: yes, because ...; moveable property: yes ...".
    """

    model_config = ConfigDict(extra="forbid")

    element: str
    satisfied: Literal["yes", "no", "unclear"]
    justification: str = ""
    supporting_provenance: list[Provenance] = Field(default_factory=list)


class SectionCandidate(BaseModel):
    """A statute the system proposes, with score, provenance, and IPC lineage."""

    model_config = ConfigDict(extra="forbid")

    act: Act
    section: str
    description_en: str = ""
    score: float = Field(ge=0.0, le=1.0)
    cognizable: bool | None = None
    elements: list[ElementCheck] = Field(default_factory=list)
    ipc_source: str | None = Field(
        default=None, description="original IPC section if this came through the IPC->BNS map"
    )
    mapping_ambiguous: bool = False


# ---------------------------------------------------------------------------
# top-level blocks
# ---------------------------------------------------------------------------


class Jurisdiction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: Literal["Tamil Nadu"] = "Tamil Nadu"
    district: ExtractedField = Field(default_factory=_empty)
    police_station: ExtractedField = Field(default_factory=_empty)
    ps_code: str | None = None


class Occurrence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    from_datetime: ExtractedField = Field(default_factory=_empty)
    to_datetime: ExtractedField = Field(default_factory=_empty)
    is_interval: bool = False
    info_received_at_ps: ExtractedField = Field(default_factory=_empty)
    gd_reference: ExtractedField = Field(default_factory=_empty)


class PlaceOfOccurrence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    direction_from_ps: ExtractedField = Field(default_factory=_empty)
    distance_from_ps_km: ExtractedField = Field(default_factory=_empty)
    beat_no: ExtractedField = Field(default_factory=_empty)
    address: ExtractedField = Field(default_factory=_empty)
    outside_ps_limits: bool = False
    other_ps_name: ExtractedField = Field(default_factory=_empty)
    other_district: ExtractedField = Field(default_factory=_empty)


class GroundingEntry(BaseModel):
    """Per-sentence faithfulness record for the narrative."""

    model_config = ConfigDict(extra="forbid")

    sentence_id: str
    supported: bool
    supporting_provenance: list[Provenance] = Field(default_factory=list)


class Narrative(BaseModel):
    """The one composed field. Every sentence must be grounded or the record
    does not pass the faithfulness gate (see `FirRecord.narrative_is_grounded`)."""

    model_config = ConfigDict(extra="forbid")

    ta: str | None = None
    en: str | None = None
    grounding: list[GroundingEntry] = Field(default_factory=list)


CognizabilityDecision = Literal["cognizable", "non_cognizable", "mixed", "undetermined"]
CognizabilityRoute = Literal["FIR", "CSR_advisory", "officer_review"]


class Cognizability(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: CognizabilityDecision = "undetermined"
    route: CognizabilityRoute = "officer_review"
    schedule_ref: str | None = None
    rationale: str = ""


RecordStatus = Literal["draft", "under_review", "verified", "csr_advisory"]


class FirRecord(BaseModel):
    """One TN FIR (CCTNS IF-1) record. `schema_version` is pinned."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["if1/v1"] = "if1/v1"
    record_id: str = Field(default_factory=lambda: f"fir-{uuid4().hex[:12]}")
    status: RecordStatus = "draft"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    jurisdiction: Jurisdiction = Field(default_factory=Jurisdiction)
    fir_number: str | None = None
    fir_year: int | None = None
    fir_date: str | None = Field(default=None, description="ISO date; None until registered")

    acts_sections: list[SectionCandidate] = Field(default_factory=list)
    occurrence: Occurrence = Field(default_factory=Occurrence)
    information_type: Literal["written", "oral"] | None = None
    place_of_occurrence: PlaceOfOccurrence = Field(default_factory=PlaceOfOccurrence)

    complainant: Person = Field(default_factory=Person)
    accused: list[Accused] = Field(default_factory=list)
    delay_reason: ExtractedField = Field(default_factory=_empty)
    properties_involved: list[PropertyItem] = Field(default_factory=list)
    total_property_value_inr: ExtractedField = Field(default_factory=_empty)
    inquest_ud_case_no: ExtractedField = Field(default_factory=_empty)
    witnesses: list[Person] = Field(default_factory=list)

    narrative: Narrative = Field(default_factory=Narrative)
    cognizability: Cognizability = Field(default_factory=Cognizability)

    audit_ref: str | None = None
    document_version: int = Field(default=1, ge=1)

    # -- invariants ---------------------------------------------------------
    @property
    def narrative_is_grounded(self) -> bool:
        """The faithfulness gate: a narrative passes only if it has text, has a
        grounding entry per sentence, and every entry is supported."""
        if not (self.narrative.ta or self.narrative.en):
            return False
        if not self.narrative.grounding:
            return False
        return all(g.supported for g in self.narrative.grounding)

    @property
    def is_fileable(self) -> bool:
        """Never True for a machine-produced record. The officer's verification
        flips `status` to 'verified'; nothing in this codebase does."""
        return self.status == "verified"

    def machine_filled_fields(self) -> list[str]:
        """Dotted paths of every slot the system filled, for the officer UI's
        'review these' list."""
        out: list[str] = []

        def walk(obj: Any, path: str) -> None:
            if isinstance(obj, ExtractedField):
                if obj.is_machine_filled:
                    out.append(path)
            elif isinstance(obj, BaseModel):
                for name in type(obj).model_fields:
                    walk(getattr(obj, name), f"{path}.{name}" if path else name)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    walk(item, f"{path}[{i}]")

        walk(self, "")
        return out


__all__ = [
    "Accused",
    "Act",
    "Bilingual",
    "Cognizability",
    "ElementCheck",
    "ExtractedField",
    "FieldStatus",
    "FirRecord",
    "GroundingEntry",
    "Jurisdiction",
    "Narrative",
    "Occurrence",
    "Passport",
    "Person",
    "PlaceOfOccurrence",
    "PropertyItem",
    "Provenance",
    "SectionCandidate",
]
