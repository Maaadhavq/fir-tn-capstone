"""Build data/statutes/bns_2023.jsonl -- the text of every BNS 2023 section.

Source: The Gazette of India Extraordinary, Part II Sec. 1, No. 53, 25 Dec 2023,
CG-DL-E-25122023-250883 -- The Bharatiya Nyaya Sanhita, 2023 (Act 45 of 2023),
English text, 102 pp. MHA copy:
https://www.mha.gov.in/sites/default/files/250883_english_01042024.pdf
(saved as data/statutes/mha_250883_english.pdf; gitignored, 1.3 MB).

Layout, and how it is read:
  * body text is 10 pt in a single column, x in [110, 480];
  * a section starts with its number as the first word of an indented line
    ("100." at x ~142) -- sub-sections "(2)" continue the same section;
  * the section *heading* is the 8 pt marginal note beside the first lines
    (left margin on even pages, right margin on odd), e.g. "Culpable homicide.";
  * chapter headings are centred 10 pt lines beginning "CHAPTER"; the small-caps
    title follows on the next line(s).

Output, one JSON object per line:
  {"section": "100", "heading": "Culpable homicide.", "chapter": "VI",
   "chapter_title": "OF OFFENCES AFFECTING THE HUMAN BODY",
   "text": "Whoever causes death ...", "gazette_page": 31}

`text` is the whole section as printed, illustrations and explanations
included; `operative_text` is the part before the first "Illustrations." line
when there is one. Numbering in the gazette is 1..358 with no gaps; the build
fails if the parse does not reproduce that.

Run:  python scripts/build_bns_text.py [--pdf PATH]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

PDF = REPO / "data" / "statutes" / "mha_250883_english.pdf"
OUT = REPO / "data" / "statutes" / "bns_2023.jsonl"

BODY_X0, BODY_X1 = 110.0, 482.0     # main column
BODY_MIN_SIZE = 9.0                 # body is 10 pt; margin notes 8 pt; small caps 7 pt
HEADER_MAX_TOP = 75.0               # running head + rule
LINE_TOL = 2.5
N_SECTIONS = 358

_SECTION_START = re.compile(r"^(\d{1,3})\.$")
_SECTION_START_GLUED = re.compile(r"^(\d{1,3})\.(\(|[A-Z])")   # "1.(1)This" on p.1 / "357.Whoever" on p.102
_CHAPTER = re.compile(r"^CHAPTER\s*([IVXL]+)$")
_ACT_CITATION = re.compile(r"^\d{1,3} of \d{4}\.?$")
_ACT_CITATION_INLINE = re.compile(r"\b\d{1,3} of \d{4}\.?\s*")
_SIGNATURE = re.compile(r"\s[A-Z][A-Z. ]{3,}, (?:Joint |Addl\. )?Secretary")


@dataclass
class Section:
    number: int
    page: int
    heading_words: list[str] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    chapter: str = ""
    chapter_title: str = ""

    def text(self) -> str:
        return _tidy(" ".join(self.lines))


def _tidy(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+([,.;:])", r"\1", s)
    s = re.sub(r"(\w)- (\w)", r"\1-\2", s)          # line-wrap hyphen: "Non- appearance"
    s = s.replace("––", "—").replace("--", "—")
    return s


def _lines(words):
    words = sorted(words, key=lambda w: (round(w["top"]), w["x0"]))
    out = []
    for w in words:
        if out and abs(w["top"] - out[-1][0]) <= LINE_TOL:
            out[-1][1].append(w)
        else:
            out.append([w["top"], [w]])
    return [(t, sorted(ws, key=lambda w: w["x0"])) for t, ws in out]


def parse(pdf_path: Path) -> list[Section]:
    import pdfplumber

    sections: list[Section] = []
    chapter, chapter_title = "", ""
    pending_chapter_title = False
    with pdfplumber.open(str(pdf_path)) as pdf:
        for pno, page in enumerate(pdf.pages, 1):
            words = [w for w in page.extract_words(x_tolerance=1.0, extra_attrs=["size"])
                     if w["top"] > HEADER_MAX_TOP]
            body = [w for w in words if BODY_X0 <= w["x0"] < BODY_X1 and w["size"] >= BODY_MIN_SIZE]
            caps = [w for w in words if BODY_X0 <= w["x0"] < BODY_X1 and w["size"] < BODY_MIN_SIZE]
            margin = [w for w in words if (w["x0"] < BODY_X0 or w["x0"] >= BODY_X1) and w["size"] < BODY_MIN_SIZE]
            # nothing below the last body line is a marginal note (the signature
            # block of the e-gazette sits under the text on the final page)
            if body and pno == len(pdf.pages):
                floor = max(w["top"] for w in body) + 12
                margin = [w for w in margin if w["top"] <= floor]

            # margin notes -> attach to the section whose first line they sit beside.
            # Collected per page, assigned after the body pass below.
            margin_lines = _lines(margin)
            starts_on_page: list[tuple[float, Section]] = []
            carry = sections[-1] if sections else None     # section current at the top of this page

            for top, ws in _lines(body + caps):
                texts = [w["text"] for w in ws]
                first = texts[0]
                line = " ".join(texts)
                m = _CHAPTER.match(line.replace(" ", "")) if "CHAPTER" in line.replace(" ", "") else None
                if m:
                    chapter, chapter_title, pending_chapter_title = m.group(1), "", True
                    continue
                if pending_chapter_title:
                    # small-caps title: "O" + "F OFFENCES ..." -> "OF OFFENCES ..."
                    t = re.sub(r"\b([A-Z]) ([A-Z])", r"\1\2", line.upper())
                    if ws[0]["size"] < BODY_MIN_SIZE or (len(ws) > 1 and ws[1]["size"] < BODY_MIN_SIZE) or line.isupper():
                        chapter_title = (chapter_title + " " + t).strip()
                        continue
                    pending_chapter_title = False
                sm = _SECTION_START.match(first)
                gm = None if sm else _SECTION_START_GLUED.match(first)
                if sm or gm:
                    num = int((sm or gm).group(1))
                    expected = sections[-1].number + 1 if sections else 1
                    if num == expected:
                        sec = Section(number=num, page=pno, chapter=chapter, chapter_title=chapter_title)
                        sections.append(sec)
                        starts_on_page.append((top, sec))
                        rest = texts[1:] if sm else [first[len(gm.group(1)) + 1:], *texts[1:]]
                        sec.lines.append(" ".join(rest))
                        continue
                if sections:
                    sections[-1].lines.append(line)

            # each margin line belongs to the nearest section start at or above it;
            # marginal *citations* of other Acts ("28 of 1961.") are not headings
            for top, ws in margin_lines:
                if _ACT_CITATION.match(" ".join(w["text"] for w in ws)):
                    continue
                cands = [(t, s) for t, s in starts_on_page if t <= top + 4]
                if cands:
                    cands[-1][1].heading_words.extend(w["text"] for w in ws)
                elif carry is not None:
                    # above the first start on this page: the heading of the
                    # section that ran over from the previous page
                    carry.heading_words.extend(w["text"] for w in ws)
    return sections


_ILLUSTRATION_HEAD = re.compile(r"\bIllustrations?\.")
_ILLUSTRATION_END = re.compile(r"\(\d{1,2}\)\s|\bExplanation\b|\bException\b|\bProvided\b")


def strip_illustrations(text: str) -> str:
    """The section without its worked examples.

    An illustration block runs from "Illustrations." to the next sub-section
    "(2)", Explanation, Exception or proviso; everything the block contains is
    lettered examples ("(a) A shoots Z ..."). What remains is the operative
    wording -- what an element check should be read against."""
    out, pos = [], 0
    for m in _ILLUSTRATION_HEAD.finditer(text):
        if m.start() < pos:
            continue
        out.append(text[pos:m.start()])
        e = _ILLUSTRATION_END.search(text, m.end())
        pos = e.start() if e else len(text)
    out.append(text[pos:])
    return _tidy(" ".join(out))


def to_records(sections: list[Section]) -> list[dict]:
    recs = []
    for s in sections:
        text = s.text()
        if s.number == N_SECTIONS:
            # the e-gazette's signature and press imprint follow the last section
            m = _SIGNATURE.search(text)
            if m:
                text = text[: m.start()].rstrip()
        recs.append({
            "section": str(s.number),
            "heading": _tidy(_ACT_CITATION_INLINE.sub("", " ".join(s.heading_words))),
            "chapter": s.chapter,
            "chapter_title": _tidy(s.chapter_title),
            "text": text,
            "operative_text": strip_illustrations(text),
            "gazette_page": s.page,
            "source": "Gazette of India Extraordinary No. 53 (25-12-2023), CG-DL-E-25122023-250883",
        })
    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, default=PDF)
    args = ap.parse_args()
    if not args.pdf.exists():
        print(f"missing {args.pdf}; download the MHA gazette PDF first (see docstring)")
        return 2
    sections = parse(args.pdf)
    nums = [s.number for s in sections]
    if nums != list(range(1, N_SECTIONS + 1)):
        missing = sorted(set(range(1, N_SECTIONS + 1)) - set(nums))
        print(f"PARSE FAILED: {len(nums)} sections, missing {missing[:20]}", file=sys.stderr)
        return 1
    recs = to_records(sections)
    with OUT.open("w", encoding="utf-8", newline="\n") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    no_heading = [r["section"] for r in recs if not r["heading"]]
    chapters = Counter(r["chapter"] for r in recs)
    print(f"{len(recs)} sections -> {OUT.relative_to(REPO)}; {len(chapters)} chapters; "
          f"{len(no_heading)} without a marginal heading {no_heading[:10]}")
    print(f"median text length {sorted(len(r['text']) for r in recs)[len(recs) // 2]} chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
