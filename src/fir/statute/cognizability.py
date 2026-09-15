"""Cognizability routing: FIR vs CSR.

Under the BNSS First Schedule each offence is classed cognizable or
non-cognizable. It decides what the officer may do:

  cognizable      -> register an FIR (BNSS s.173); police may investigate
                     without prior magistrate approval.
  non-cognizable  -> Community Service Register entry; investigation needs a
                     magistrate's order.

Getting this wrong in either direction is serious -- refusing an FIR on a
cognizable offence is itself misconduct -- so an unknown section NEVER silently
defaults. It routes to the officer.

SOURCE. `data/statutes/bnss_schedule1.csv`, parsed from the official gazette
text of the BNSS 2023 First Schedule by `scripts/build_bnss_schedule.py`, is the
table in force when it exists. The hand-coded stub below is the fallback that
kept the slice running before the CSV was built; `schedule_source()` tells the
form which one is in force so the officer is never shown stub output as if it
were authority. (Building the CSV showed the stub wrong on 4 of 48 entries --
BNS 126, 223, 296, 329 -- which is why the stub is not authority.)

CONDITIONAL entries. The Schedule does not always give a flat answer:
"According as offence abetted is cognizable or non-cognizable" (abetment,
conspiracy, attempt), "Cognizable if information ... is given by the person
aggrieved" (s.85 cruelty), or a split by property value (s.303(2) theft:
non-cognizable under Rs 5,000). Those are `conditional`, route to the officer
like `unknown`, and carry the Schedule's own words in the rationale. Where the
condition is on a fact the pipeline extracts (the property value) a
machine-readable `condition` resolves it when that fact is present.
"""

from __future__ import annotations

import csv
import functools
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from fir.config import DATA_DIR


class Route(str, Enum):
    """Where a case goes after statute identification."""

    FIR = "FIR"
    CSR = "CSR"
    OFFICER_REVIEW = "OFFICER_REVIEW"


class Cognizability(str, Enum):
    COGNIZABLE = "cognizable"
    NON_COGNIZABLE = "non_cognizable"
    #: the Schedule makes it depend on a fact (who reports, value, the offence
    #: abetted); the words of the condition are in `Schedule.note_for`
    CONDITIONAL = "conditional"
    #: not in the table at all
    UNKNOWN = "unknown"


#: the two values that do not settle the route on their own
UNSETTLED = frozenset({Cognizability.CONDITIONAL, Cognizability.UNKNOWN})


# BNS section (base number, sub-clauses stripped) -> cognizability.
# STUB -- see module docstring. Keyed on the base number because the Schedule
# classes offences at that level for everything we cover here.
_STUB_SCHEDULE: dict[str, Cognizability] = {
    # --- offences against the body ---
    "100": Cognizability.COGNIZABLE,   # culpable homicide (definition)
    "101": Cognizability.COGNIZABLE,   # murder (definition)
    "103": Cognizability.COGNIZABLE,   # punishment for murder
    "105": Cognizability.COGNIZABLE,   # culpable homicide not amounting to murder
    "106": Cognizability.COGNIZABLE,   # death by negligence
    "108": Cognizability.COGNIZABLE,   # abetment of suicide
    "109": Cognizability.COGNIZABLE,   # attempt to murder
    "110": Cognizability.COGNIZABLE,   # attempt to culpable homicide
    "115": Cognizability.NON_COGNIZABLE,  # voluntarily causing hurt
    "117": Cognizability.COGNIZABLE,   # grievous hurt
    "118": Cognizability.COGNIZABLE,   # hurt by dangerous weapon
    # --- offences against women / children ---
    "64": Cognizability.COGNIZABLE,    # rape
    "74": Cognizability.COGNIZABLE,    # assault to outrage modesty
    "75": Cognizability.COGNIZABLE,    # sexual harassment
    "80": Cognizability.COGNIZABLE,    # dowry death
    "85": Cognizability.COGNIZABLE,    # cruelty by husband/relatives
    "87": Cognizability.COGNIZABLE,    # kidnapping to compel marriage
    # --- liberty ---
    "126": Cognizability.NON_COGNIZABLE,  # wrongful restraint  (Schedule: cognizable)
    "127": Cognizability.COGNIZABLE,   # wrongful confinement
    "137": Cognizability.COGNIZABLE,   # kidnapping
    "140": Cognizability.COGNIZABLE,   # kidnapping for ransom etc.
    # --- property ---
    "303": Cognizability.COGNIZABLE,   # theft
    "304": Cognizability.COGNIZABLE,   # snatching
    "305": Cognizability.COGNIZABLE,   # theft in dwelling
    "309": Cognizability.COGNIZABLE,   # robbery
    "310": Cognizability.COGNIZABLE,   # dacoity
    "316": Cognizability.COGNIZABLE,   # criminal breach of trust
    "318": Cognizability.COGNIZABLE,   # cheating
    "324": Cognizability.NON_COGNIZABLE,  # mischief (varies by damage value)
    "326": Cognizability.COGNIZABLE,   # mischief by fire/explosive
    "329": Cognizability.NON_COGNIZABLE,  # criminal trespass  (Schedule: cognizable)
    "331": Cognizability.COGNIZABLE,   # house-breaking
    "332": Cognizability.COGNIZABLE,   # house-trespass
    # --- documents / public order ---
    "61": Cognizability.COGNIZABLE,    # criminal conspiracy
    "189": Cognizability.COGNIZABLE,   # unlawful assembly
    "190": Cognizability.COGNIZABLE,   # common object
    "191": Cognizability.COGNIZABLE,   # rioting
    "221": Cognizability.NON_COGNIZABLE,  # obstructing public servant
    "223": Cognizability.NON_COGNIZABLE,  # disobedience to order  (Schedule: cognizable)
    "238": Cognizability.COGNIZABLE,   # causing disappearance of evidence
    "281": Cognizability.COGNIZABLE,   # rash driving
    "296": Cognizability.NON_COGNIZABLE,  # obscene acts  (Schedule: cognizable)
    "336": Cognizability.NON_COGNIZABLE,  # forgery
    "351": Cognizability.NON_COGNIZABLE,  # criminal intimidation
    "352": Cognizability.NON_COGNIZABLE,  # intentional insult
    "356": Cognizability.NON_COGNIZABLE,  # defamation
    # --- general clauses (no independent cognizability) ---
    "3": Cognizability.UNKNOWN,        # common intention -- follows the main offence
    "45": Cognizability.UNKNOWN,       # abetment -- follows the main offence
    "49": Cognizability.UNKNOWN,
}

_BNS_NUM_RE = re.compile(r"BNS\s*([0-9]+)")
_BASE_RE = re.compile(r"^(\d+)")
_CLAUSE_RE = re.compile(r"\([0-9a-z]+\)")

# ---------------------------------------------------------------------------
# schedule source: the real table if present, the stub otherwise
# ---------------------------------------------------------------------------
#
# data/statutes/bnss_schedule1.csv, when it exists, replaces the stub above
# entirely. Columns (header row required; extra columns are ignored):
#
#   bns_section  base number ("103") or sub-clause ("103(1)"); a sub-clause row
#                overrides the base row for that exact clause
#   cognizable   cognizable | non_cognizable | conditional | unknown
#                (case-insensitive; Y/N accepted)
#   condition    optional machine-readable rule for a `conditional` row, e.g.
#                "property_value_inr<5000=non_cognizable;else=cognizable"
#   notes        the Schedule's own words for a conditional entry, or why a
#                base row is unresolved -- shown to the officer
#   source_ref   gazette page / notification so a reviewer can check the row
#
# Nothing here is legal authority beyond the words of the Schedule; the stub
# is a placeholder and `schedule_source()` says so when it is in force.

_SCHEDULE_PATH_PARTS = ("statutes", "bnss_schedule1.csv")
_TRUTHY = {"y", "yes", "true", "1", "cognizable", "c"}
_FALSY = {"n", "no", "false", "0", "non_cognizable", "non-cognizable", "nc"}
_COND_CLAUSE_RE = re.compile(r"^(\w+)\s*(<=|>=|<|>|==)\s*(-?\d+(?:\.\d+)?)$")


@dataclass(slots=True)
class Schedule:
    """Cognizability lookup with provenance of where the answers came from."""

    table: dict[str, Cognizability]
    source: str                      # "stub" or the CSV path
    n_rows: int
    source_refs: dict[str, str] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)
    conditions: dict[str, str] = field(default_factory=dict)
    derived: set[str] = field(default_factory=set)   # base rows inferred from sub-clauses

    @property
    def is_stub(self) -> bool:
        return self.source == "stub"

    # -- keys ----------------------------------------------------------------
    @staticmethod
    def _normalise(bns_section: str) -> str:
        return bns_section.upper().replace("BNS", "").replace(" ", "").strip()

    @classmethod
    def _expand(cls, key: str) -> list[str]:
        """'324(4),(5)' -> ['324(4)', '324(5)']; '103(1)' -> ['103(1)']."""
        if "," not in key:
            return [key]
        m = _BASE_RE.match(key)
        if not m:
            return [key]
        base = m.group(1)
        clauses = _CLAUSE_RE.findall(key)
        return [base + c for c in clauses] if clauses else [key]

    def _key_for(self, part: str) -> str | None:
        """The table key that answers for `part`: exact clause, else its base."""
        if part in self.table:
            return part
        m = _BASE_RE.match(part)
        if m and m.group(1) in self.table:
            return m.group(1)
        return None

    # -- lookup --------------------------------------------------------------
    def lookup(self, bns_section: str, context: dict | None = None) -> Cognizability:
        """Exact clause first ('103(1)'), then the base section ('103').

        A composite target ('324(4),(5)') is the answer its parts agree on, or
        UNKNOWN. A `conditional` entry is resolved through its `condition`
        when `context` carries the fact it needs (e.g. property_value_inr).
        """
        parts = self._expand(self._normalise(bns_section))
        values: set[Cognizability] = set()
        for part in parts:
            key = self._key_for(part)
            if key is None:
                values.add(Cognizability.UNKNOWN)
                continue
            v = self.table[key]
            if v is Cognizability.CONDITIONAL and context and key in self.conditions:
                resolved = _eval_condition(self.conditions[key], context)
                if resolved is not None:
                    v = resolved
            values.add(v)
        return values.pop() if len(values) == 1 else Cognizability.UNKNOWN

    def note_for(self, bns_section: str) -> str:
        """The Schedule's words behind a conditional/unresolved entry, if any."""
        notes = []
        for part in self._expand(self._normalise(bns_section)):
            key = self._key_for(part)
            if key is not None and self.notes.get(key):
                notes.append(self.notes[key])
        return " | ".join(dict.fromkeys(notes))


def _eval_condition(cond: str, context: dict) -> Cognizability | None:
    """Evaluate "field<5000=non_cognizable;else=cognizable" against `context`.

    Returns None when the fact the condition needs is not in the context --
    the entry then stays CONDITIONAL and goes to the officer.
    """
    for clause in filter(None, (c.strip() for c in cond.split(";"))):
        if "=" not in clause:
            continue
        expr, _, outcome = clause.rpartition("=")
        expr = expr.strip()
        out = _parse_cog(outcome)
        if expr == "else":
            return out
        m = _COND_CLAUSE_RE.match(expr)
        if not m:
            return None
        fact, op, threshold = m.groups()
        val = context.get(fact)
        if val is None:
            return None
        try:
            x, t = float(val), float(threshold)
        except (TypeError, ValueError):
            return None
        hit = {"<": x < t, "<=": x <= t, ">": x > t, ">=": x >= t, "==": x == t}[op]
        if hit:
            return out
    return None


def _parse_cog(v: str) -> Cognizability:
    t = (v or "").strip().lower()
    if t in _TRUTHY:
        return Cognizability.COGNIZABLE
    if t in _FALSY:
        return Cognizability.NON_COGNIZABLE
    if t == "conditional":
        return Cognizability.CONDITIONAL
    return Cognizability.UNKNOWN


def _load_csv(path: Path) -> Schedule:
    table: dict[str, Cognizability] = {}
    refs: dict[str, str] = {}
    notes: dict[str, str] = {}
    conds: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            sec = Schedule._normalise(row.get("bns_section") or "")
            if not sec:
                continue
            table[sec] = _parse_cog(row.get("cognizable", ""))
            if row.get("source_ref"):
                refs[sec] = row["source_ref"].strip()
            if row.get("notes"):
                notes[sec] = row["notes"].strip()
            if row.get("condition"):
                conds[sec] = row["condition"].strip()
    n_rows = len(table)

    # Base rows the Schedule does not print: '103' from 103(1)/103(2). Only
    # where every sub-clause agrees; otherwise the base is UNKNOWN with the
    # disagreement in its note, so 'BNS 324' asks the officer which clause.
    by_base: dict[str, list[str]] = defaultdict(list)
    for key in table:
        m = _BASE_RE.match(key)
        if m and m.group(1) != key:
            by_base[m.group(1)].append(key)
    derived: set[str] = set()
    for base, keys in by_base.items():
        if base in table:
            continue
        vals = {table[k] for k in keys}
        derived.add(base)
        if len(vals) == 1:
            table[base] = vals.pop()
            if any(k in notes for k in keys):
                notes[base] = " | ".join(dict.fromkeys(notes[k] for k in keys if k in notes))
            if len(keys) == 1 and keys[0] in conds:
                conds[base] = conds[keys[0]]
        else:
            table[base] = Cognizability.UNKNOWN
            notes[base] = "sub-clauses differ: " + ", ".join(
                f"{k} {table[k].value}" for k in sorted(keys))
        refs[base] = "derived from " + ", ".join(sorted(keys))
    return Schedule(table=table, source=str(path), n_rows=n_rows, source_refs=refs,
                    notes=notes, conditions=conds, derived=derived)


@functools.lru_cache(maxsize=1)
def schedule() -> Schedule:
    """The active schedule. Loads the CSV if it exists, else the stub."""
    path = DATA_DIR.joinpath(*_SCHEDULE_PATH_PARTS)
    if path.exists():
        return _load_csv(path)
    return Schedule(table=dict(_STUB_SCHEDULE), source="stub", n_rows=len(_STUB_SCHEDULE))


def schedule_source() -> str:
    """Human-readable provenance string for the form's 'Basis' line."""
    s = schedule()
    if s.is_stub:
        return (
            f"BNSS 2023 First Schedule -- STUB ({s.n_rows} common offences, hand-coded, "
            "not the official table; see cognizability.py)"
        )
    return f"BNSS 2023 First Schedule -- {Path(s.source).name} ({s.n_rows} rows)"


@dataclass(slots=True)
class CognizabilityResult:
    """Routing decision plus the per-section evidence behind it."""

    route: Route
    per_section: dict[str, Cognizability] = field(default_factory=dict)
    unknown_sections: list[str] = field(default_factory=list)
    rationale: str = ""

    @property
    def cognizable_sections(self) -> list[str]:
        return [
            s for s, c in self.per_section.items() if c is Cognizability.COGNIZABLE
        ]

    @property
    def conditional_sections(self) -> list[str]:
        return [
            s for s, c in self.per_section.items() if c is Cognizability.CONDITIONAL
        ]


def base_section(bns_section: str) -> str | None:
    """'BNS 103(1)' -> '103'. Returns None if no BNS number is present."""
    m = _BNS_NUM_RE.search(bns_section.upper())
    return m.group(1) if m else None


def classify_section(bns_section: str, context: dict | None = None) -> Cognizability:
    """Look up one BNS section in the active Schedule (CSV if present, else stub)."""
    return schedule().lookup(bns_section, context)


def route_case(
    bns_sections: list[str],
    has_review_items: bool = False,
    context: dict | None = None,
) -> CognizabilityResult:
    """Decide FIR / CSR / OFFICER_REVIEW for a set of BNS sections.

    `context` carries extracted facts a conditional Schedule entry may turn on
    (today: `property_value_inr`).

    Policy, in order:

    1. Any cognizable section  -> FIR. Cognizable dominates: a case carrying
       both classes is registered as an FIR.
    2. No sections at all      -> OFFICER_REVIEW.
    3. Any unknown or conditional section, or unresolved items in the mapping
       review queue           -> OFFICER_REVIEW, because the answer could still
                                  become FIR once the open point is resolved.
    4. All non-cognizable      -> CSR.
    """
    sch = schedule()
    per_section = {s: sch.lookup(s, context) for s in bns_sections}
    unknown = [s for s, c in per_section.items() if c is Cognizability.UNKNOWN]
    conditional = [s for s, c in per_section.items() if c is Cognizability.CONDITIONAL]

    if any(c is Cognizability.COGNIZABLE for c in per_section.values()):
        cog = [s for s, c in per_section.items() if c is Cognizability.COGNIZABLE]
        return CognizabilityResult(
            route=Route.FIR,
            per_section=per_section,
            unknown_sections=unknown + conditional,
            rationale=f"cognizable offence(s) present: {', '.join(cog)}",
        )

    if not bns_sections:
        return CognizabilityResult(
            route=Route.OFFICER_REVIEW,
            per_section=per_section,
            unknown_sections=unknown,
            rationale="no BNS section could be auto-applied",
        )

    if unknown or conditional or has_review_items:
        why = []
        if conditional:
            why.append("cognizability is conditional: " + "; ".join(
                f"{s} ({sch.note_for(s) or 'see Schedule'})" for s in conditional))
        if unknown:
            why.append("unclassified section(s): " + ", ".join(
                f"{s} ({sch.note_for(s)})" if sch.note_for(s) else s for s in unknown))
        if has_review_items:
            why.append("unresolved IPC->BNS review items")
        return CognizabilityResult(
            route=Route.OFFICER_REVIEW,
            per_section=per_section,
            unknown_sections=unknown + conditional,
            rationale="; ".join(why),
        )

    return CognizabilityResult(
        route=Route.CSR,
        per_section=per_section,
        unknown_sections=unknown,
        rationale="all identified offences are non-cognizable",
    )
