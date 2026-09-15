"""The words of the BNS 2023 sections, from the gazette text.

`data/statutes/bns_2023.jsonl` (built by `scripts/build_bns_text.py` from the
official gazette PDF) has one record per section: heading, chapter, full text,
and `operative_text` (the section without its worked illustrations). This
module is the read side: given a BNS target such as "BNS 303(2)" it returns
the section, and the sub-section paragraph when the target names one.

Why it exists: the classifier is trained on IPC labels and `statute_text.py`
shows the *IPC* wording ILSI ships. The officer files under the BNS, so the
suggestion must show the BNS words -- the IPC text is lineage, not authority.
The file is optional: without it every accessor returns None and the decision
record simply has no BNS text, which the form says.
"""

from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass
from pathlib import Path

from fir.config import DATA_DIR

PATH_PARTS = ("statutes", "bns_2023.jsonl")

_BASE_RE = re.compile(r"(\d{1,3})")
_CLAUSE_RE = re.compile(r"\((\d{1,2})\)")
_SUBSECTION_SPLIT = re.compile(r"(?=\(\d{1,2}\)\s)")


@dataclass(frozen=True, slots=True)
class BnsSection:
    section: str
    heading: str
    chapter: str
    chapter_title: str
    text: str
    operative_text: str
    gazette_page: int

    def clause(self, n: str) -> str | None:
        """The sub-section paragraph "(n) ..." of the operative text, if printed."""
        for part in _SUBSECTION_SPLIT.split(self.operative_text):
            if part.startswith(f"({n}) "):
                return part.strip()
        return None


class BnsText:
    def __init__(self, records: list[dict]):
        self._by_section = {
            r["section"]: BnsSection(
                section=r["section"], heading=r.get("heading", ""), chapter=r.get("chapter", ""),
                chapter_title=r.get("chapter_title", ""), text=r.get("text", ""),
                operative_text=r.get("operative_text") or r.get("text", ""),
                gazette_page=int(r.get("gazette_page") or 0),
            )
            for r in records
        }

    @classmethod
    def path(cls) -> Path:
        return DATA_DIR.joinpath(*PATH_PARTS)

    @classmethod
    @functools.lru_cache(maxsize=1)
    def load(cls) -> "BnsText | None":
        p = cls.path()
        if not p.exists():
            return None
        with p.open(encoding="utf-8") as fh:
            return cls([json.loads(line) for line in fh if line.strip()])

    def __len__(self) -> int:
        return len(self._by_section)

    # -- lookup ---------------------------------------------------------------
    @staticmethod
    def base_of(bns_section: str) -> str | None:
        m = _BASE_RE.search(bns_section.upper().replace("BNS", ""))
        return m.group(1) if m else None

    def section(self, bns_section: str) -> BnsSection | None:
        base = self.base_of(bns_section)
        return self._by_section.get(base) if base else None

    def text_for(self, bns_section: str) -> str | None:
        """Operative text of the sub-section named by the target ("303(2)"),
        falling back to the whole section's operative text. A composite target
        ("324(4),(5)") gets each named clause in turn."""
        sec = self.section(bns_section)
        if sec is None:
            return None
        clauses = _CLAUSE_RE.findall(bns_section.split(",")[0]) + [
            c for part in bns_section.split(",")[1:] for c in _CLAUSE_RE.findall(part)
        ]
        if clauses:
            found = [t for t in (sec.clause(c) for c in clauses) if t]
            if found:
                return " ".join(found)
        return sec.operative_text

    def heading_for(self, bns_section: str) -> str | None:
        sec = self.section(bns_section)
        return sec.heading if sec else None


def bns_text() -> BnsText | None:
    """The loaded corpus, or None when the jsonl has not been built."""
    return BnsText.load()
