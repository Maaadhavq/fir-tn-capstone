"""IPC 1860 -> BNS 2023 section conversion.

Why this exists: our statute-ID training data (ILSI) is IPC-labelled, but police
have filed under the BNS since 1 July 2024. The classifier predicts IPC; this
module converts to BNS.

**The safety rule** (data/mapping/README.md): the mapping table was built from a
third-party pocket directory, *not* the official MHA gazette. Only rows marked
`confidence == high` AND `needs_review == N` may be applied without a human.
Everything else goes to a review queue with an explicit reason.

Beyond the confidence flags, four classes of row are not plain BNS sections and
are handled separately -- auto-applying any of them would put a wrong section on
a real FIR:

  OMITTED       IPC 13 -- repealed in 1950, no BNS equivalent.
  PC Act 1988   IPC 161/164 -- corruption offences moved to the Prevention of
                Corruption Act, a *different statute*. Filing these under BNS
                would be legally wrong, so they are always routed out.
  VERIFY        IPC 155/156/190/482 -- placeholder, mapping not yet established.
  unprefixed    IPC 427 stores a bare "324(4),(5)" -- a formatting slip in the
                CSV; we normalise it rather than silently dropping the row.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from fir.config import DATA_DIR, pipeline, resolve

DEFAULT_MAP_PATH = DATA_DIR / "mapping" / "ipc_bns_map.csv"

#: Values in the bns_section column that are not BNS sections at all.
SENTINEL_OMITTED = "OMITTED"
SENTINEL_VERIFY = "VERIFY"
OTHER_STATUTE_MARKERS = ("PC Act",)


class ReviewReason(str, Enum):
    """Why a predicted IPC section did not get an automatic BNS section."""

    LOW_CONFIDENCE = "low_confidence"
    FLAGGED_NEEDS_REVIEW = "flagged_needs_review"
    MAPPING_UNVERIFIED = "mapping_unverified"
    OTHER_STATUTE = "maps_to_other_statute"
    REPEALED_NO_EQUIVALENT = "repealed_no_equivalent"
    NOT_IN_TABLE = "not_in_mapping_table"
    CUES_PRESENT_NOT_PREDICTED = "cues_present_not_predicted"   # raised by the element cue scan


@dataclass(frozen=True, slots=True)
class MapRow:
    """One row of ipc_bns_map.csv."""

    ipc_section: str
    offence: str
    bns_section: str
    confidence: str
    needs_review: bool
    notes: str = ""

    @property
    def is_bns_section(self) -> bool:
        """True if the target is an actual BNS section, not a sentinel."""
        v = self.bns_section.strip()
        if not v or v in (SENTINEL_OMITTED, SENTINEL_VERIFY):
            return False
        return not any(m in v for m in OTHER_STATUTE_MARKERS)

    @property
    def normalised_bns(self) -> str:
        """The BNS section with its prefix guaranteed.

        Repairs the one CSV row (IPC 427) that omits it.
        """
        v = self.bns_section.strip()
        return v if v.upper().startswith("BNS ") else "BNS " + v


@dataclass(frozen=True, slots=True)
class ReviewItem:
    """A prediction a human must adjudicate before it can be used."""

    ipc_section: str
    reason: ReviewReason
    offence: str = ""
    suggested_bns: str | None = None
    confidence: str | None = None
    notes: str = ""


@dataclass(slots=True)
class MappingResult:
    """Outcome of converting a set of predicted IPC sections to BNS."""

    bns_sections: list[str] = field(default_factory=list)
    auto_applied: list[MapRow] = field(default_factory=list)
    review_queue: list[ReviewItem] = field(default_factory=list)

    @property
    def n_auto(self) -> int:
        return len(self.auto_applied)

    @property
    def n_review(self) -> int:
        return len(self.review_queue)

    def summary(self) -> str:
        auto = ", ".join(self.bns_sections) or "none"
        return f"{self.n_auto} auto-applied ({auto}); {self.n_review} to review"


class IpcBnsMap:
    """Lookup table with the auto-apply policy baked in."""

    def __init__(self, rows: dict[str, MapRow], auto_apply_confidence: set[str]):
        self._rows = rows
        self._auto_conf = auto_apply_confidence

    # -- construction -------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path | None = None) -> "IpcBnsMap":
        cfg = pipeline().get("ipc_bns", {})
        path = resolve(path or cfg.get("map_path", DEFAULT_MAP_PATH))
        if not Path(path).exists():
            raise FileNotFoundError(f"IPC->BNS map not found: {path}")

        rows: dict[str, MapRow] = {}
        with open(path, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                sec = r["ipc_section"].strip().upper()
                rows[sec] = MapRow(
                    ipc_section=sec,
                    offence=r.get("offence", "").strip(),
                    bns_section=r.get("bns_section", "").strip(),
                    confidence=r.get("confidence", "").strip().lower(),
                    needs_review=r.get("needs_review", "").strip().upper() == "Y",
                    notes=r.get("notes", "").strip(),
                )
        conf = set(cfg.get("auto_apply_confidence", ["high"]))
        return cls(rows, conf)

    # -- lookup -------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._rows)

    def __contains__(self, ipc_section: str) -> bool:
        return ipc_section.strip().upper() in self._rows

    def lookup(self, ipc_section: str) -> MapRow | None:
        return self._rows.get(ipc_section.strip().upper())

    def auto_applicable(self, row: MapRow) -> bool:
        """The single source of truth for the auto-apply policy."""
        return (
            row.confidence in self._auto_conf
            and not row.needs_review
            and row.is_bns_section
        )

    # -- the operation the pipeline actually calls ---------------------------
    def apply(self, ipc_sections: list[str]) -> MappingResult:
        """Convert predicted IPC sections to BNS, splitting auto vs review.

        Order is preserved and duplicates collapse, so output is stable for
        golden-regression tests.
        """
        result = MappingResult()
        seen: set[str] = set()

        for raw in ipc_sections:
            sec = raw.strip().upper()
            if not sec or sec in seen:
                continue
            seen.add(sec)

            row = self.lookup(sec)
            if row is None:
                result.review_queue.append(
                    ReviewItem(ipc_section=sec, reason=ReviewReason.NOT_IN_TABLE)
                )
                continue

            if self.auto_applicable(row):
                result.auto_applied.append(row)
                bns = row.normalised_bns
                if bns not in result.bns_sections:
                    result.bns_sections.append(bns)
            else:
                result.review_queue.append(
                    ReviewItem(
                        ipc_section=sec,
                        reason=_review_reason(row, self._auto_conf),
                        offence=row.offence,
                        suggested_bns=row.bns_section or None,
                        confidence=row.confidence,
                        notes=row.notes,
                    )
                )
        return result

    # -- introspection, used by the harness report --------------------------
    def coverage(self) -> dict[str, int]:
        auto = sum(1 for r in self._rows.values() if self.auto_applicable(r))
        return {
            "total": len(self._rows),
            "auto_applicable": auto,
            "needs_review": sum(1 for r in self._rows.values() if r.needs_review),
            "non_bns_targets": sum(
                1 for r in self._rows.values() if not r.is_bns_section
            ),
        }


def _review_reason(row: MapRow, auto_conf: set[str]) -> ReviewReason:
    """Pick the most specific reason a row was not auto-applied.

    Statute-class problems are reported ahead of confidence problems: telling an
    officer this is a Prevention of Corruption Act offence is more actionable
    than telling them confidence was medium.
    """
    v = row.bns_section.strip()
    if v == SENTINEL_OMITTED:
        return ReviewReason.REPEALED_NO_EQUIVALENT
    if v == SENTINEL_VERIFY or not v:
        return ReviewReason.MAPPING_UNVERIFIED
    if any(m in v for m in OTHER_STATUTE_MARKERS):
        return ReviewReason.OTHER_STATUTE
    if row.needs_review:
        return ReviewReason.FLAGGED_NEEDS_REVIEW
    if row.confidence not in auto_conf:
        return ReviewReason.LOW_CONFIDENCE
    return ReviewReason.FLAGGED_NEEDS_REVIEW
