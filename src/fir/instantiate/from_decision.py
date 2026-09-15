"""Stage D, step one: carry the statute decision into the IF-1 record.

The vertical slice already produces a `StatuteDecision` (which BNS sections
were auto-applied, which were held back, and how the case routes). This module
writes that into the two IF-1 blocks it owns -- `acts_sections` and
`cognizability` -- and touches nothing else. Extraction (Stage B) fills the
particulars; the narrative composer fills `narrative`. Keeping the writers
separate is what lets each one be verified on its own.

Two deliberate choices:

* **Held-back sections are recorded, not dropped.** A review-queue item becomes
  a `SectionCandidate` with `mapping_ambiguous=True` and a low score, so the
  officer sees "the model thought 500 might apply but the BNS mapping is
  unverified" rather than nothing. Silence would hide exactly the cases that
  most need a human.

* **Route names are translated, not passed through.** The slice says
  `OFFICER_REVIEW`; the IF-1 schema says `officer_review`. The schema is the
  contract; the slice's enum is an implementation detail.
"""

from __future__ import annotations

from fir.schema.if1 import Cognizability, FirRecord, SectionCandidate
from fir.schema.statute_decision import StatuteDecision
from fir.statute.cognizability import schedule_source
from fir.statute.elements import check_elements

#: Route names: slice enum -> IF-1 schema enum.
_ROUTE = {
    "FIR": "FIR",
    "CSR": "CSR_advisory",
    "OFFICER_REVIEW": "officer_review",
}

#: Score given to a section the mapping layer refused to auto-apply. Low enough
#: to sort below every auto-applied section, non-zero so it is still visible.
_HELD_BACK_SCORE = 0.25


def _bare_section(bns: str) -> str:
    """'BNS 103(1)' -> '103(1)'. The `act` field carries the statute."""
    s = bns.strip()
    return s[4:].strip() if s.upper().startswith("BNS ") else s


def section_candidates(decision: StatuteDecision) -> list[SectionCandidate]:
    """Auto-applied sections first (ranked by classifier score where known),
    then held-back ones flagged ambiguous. Each auto-applied section carries its
    element-wise checks against the narrative (fir.statute.elements, v0)."""
    top_k = {sec: score for sec, score in decision.ipc_top_k}
    narrative = decision.narrative or ""
    out: list[SectionCandidate] = []

    for s in decision.suggested_sections:
        out.append(
            SectionCandidate(
                act="BNS_2023",
                section=_bare_section(s.bns_section),
                description_en=s.offence,
                score=float(top_k.get(s.ipc_section, 1.0)),
                cognizable=(
                    True if s.cognizable == "cognizable"
                    else False if s.cognizable == "non_cognizable"
                    else None
                ),
                elements=check_elements(s.ipc_section, narrative),
                ipc_source=s.ipc_section,
                mapping_ambiguous=False,
            )
        )

    for f in decision.review_flags:
        # Only rows that at least *suggest* a BNS target become candidates. A
        # NOT_IN_TABLE or PC-Act row has no BNS section to propose.
        if not f.suggested_bns or not f.suggested_bns.upper().startswith("BNS "):
            continue
        out.append(
            SectionCandidate(
                act="BNS_2023",
                section=_bare_section(f.suggested_bns),
                description_en=f.offence,
                score=min(_HELD_BACK_SCORE, float(top_k.get(f.ipc_section, _HELD_BACK_SCORE))),
                cognizable=None,
                ipc_source=f.ipc_section,
                mapping_ambiguous=True,
            )
        )

    out.sort(key=lambda c: (-c.score, c.mapping_ambiguous, c.section))
    return out


def cognizability_block(decision: StatuteDecision) -> Cognizability:
    """Collapse per-section cognizability into the IF-1's single decision."""
    classes = {s.cognizable for s in decision.suggested_sections}
    if not decision.suggested_sections:
        cls = "undetermined"
    elif classes == {"cognizable"}:
        cls = "cognizable"
    elif classes == {"non_cognizable"}:
        cls = "non_cognizable"
    elif classes <= {"unknown", "conditional"}:
        cls = "undetermined"
    else:
        cls = "mixed"

    return Cognizability(
        decision=cls,
        route=_ROUTE.get(decision.route, "officer_review"),
        schedule_ref=schedule_source(),
        rationale=decision.rationale,
    )


def apply_decision(record: FirRecord, decision: StatuteDecision) -> FirRecord:
    """Write the statute decision into `record`. Returns the same record.

    Sets `status` to 'csr_advisory' for a CSR route so the officer UI can
    render it as an advisory rather than an FIR draft. Never sets 'verified'.
    """
    record.acts_sections = section_candidates(decision)
    record.cognizability = cognizability_block(decision)
    if record.cognizability.route == "CSR_advisory":
        record.status = "csr_advisory"
    elif record.status == "csr_advisory":
        record.status = "draft"
    if decision.transcript is not None:
        record.information_type = "oral"
    return record


def record_from_decision(decision: StatuteDecision) -> FirRecord:
    """A fresh IF-1 draft carrying only the statute decision.

    Everything else on the form is `status='empty'` -- this is what the record
    looks like before Stage B has run.
    """
    return apply_decision(FirRecord(), decision)
