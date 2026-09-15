"""Build data/statutes/bnss_schedule1.csv from the official BNSS 2023 gazette PDF.

Source: The Gazette of India Extraordinary, Part II Sec. 1, No. 55, 25 Dec 2023,
CG-DL-E-25122023-250884 -- The Bharatiya Nagarik Suraksha Sanhita, 2023 (Act 46
of 2023), THE FIRST SCHEDULE, Part I "Offences under the Bharatiya Nyaya
Sanhita" (gazette pp. 158-188). MHA hosts the English text at
https://www.mha.gov.in/sites/default/files/2024-04/250884_2_english_01042024.pdf
(saved as data/statutes/mha_250884_english.pdf).

The Schedule is a six-column table with no ruling lines, so this is a
coordinate parser: every word is assigned to a column by its x position and
rows are cut where a section number appears in column 1 or a vertical gap
separates two entries. Two files come out:

  bnss_schedule1_rows.csv  -- one line per Schedule row, verbatim text, gazette
                              page. For the legal reviewer; nothing derived.
  bnss_schedule1.csv       -- one line per BNS section/sub-clause, the table
                              `fir.statute.cognizability` loads. Where the
                              Schedule splits a section into unlabelled
                              variants that disagree, the section is written
                              as `unknown` with the variants in `notes`, so it
                              routes to the officer rather than guessing.

Classification (column 4) is reduced conservatively:
  "Cognizable."               -> cognizable
  "Non-cognizable."           -> non_cognizable
  any other wording           -> conditional, the Schedule's words kept in `notes`
    ("According as offence abetted is cognizable...", "Cognizable if
    information ... is given by the person aggrieved ...", etc.)
  variants that disagree      -> conditional, with the variants in `notes`;
    where the split is on a fact the pipeline extracts, KNOWN_CONDITIONS below
    adds a machine-readable `condition` (today only 303(2) theft, Rs 5,000).
  empty cell                  -> unknown (a parse gap -- look at the rows file)

Run:  python scripts/build_bnss_schedule.py [--pdf PATH] [--check]
`--check` compares the result with the hand-coded stub in cognizability.py and
prints every disagreement -- each one is either a stub error or a parse error
and must be looked at by a person.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

PDF = REPO / "data" / "statutes" / "mha_250884_english.pdf"
OUT_ROWS = REPO / "data" / "statutes" / "bnss_schedule1_rows.csv"
OUT_TABLE = REPO / "data" / "statutes" / "bnss_schedule1.csv"

# gazette page numbers (1-based) of Part I of the First Schedule
FIRST_PAGE, LAST_PAGE = 158, 188
# Columns 1-3 and 6 sit at fixed x positions on every page; the two
# classification columns drift by up to 35 pt between pages (and between rows
# on pp. 158-159, 171, 175), so those two are located per row from the words
# that can only begin a classification cell.
COLS = ("section", "offence", "punishment", "cognizable", "bailable", "court")
X_OFFENCE, X_PUNISHMENT, X_COURT = 83.0, 186.0, 447.0
X_MID_MIN = 250.0             # no punishment *cell* starts right of this
COG_VOCAB = {"cognizable", "non-cognizable", "according", "non-"}
BAIL_VOCAB = {"bailable", "non-bailable", "according", "non-"}
DEFAULT_COG_START, DEFAULT_BAIL_START = 297.0, 384.0
LINE_TOL = 3.0          # words within this many pt of `top` are one line
ROW_GAP = 12.0          # a larger vertical gap can start a new (unlabelled) row

_SECTION_RE = re.compile(r"^\d{1,3}(\([0-9a-z]+\))*$")

#: Sections whose Schedule rows split on a fact the pipeline extracts. Each
#: entry is (expected variant classifications in Schedule order, condition).
#: The build asserts the parsed rows still say what the condition encodes, so
#: a changed PDF or parser cannot silently detach the rule from its source.
KNOWN_CONDITIONS: dict[str, tuple[list[str], str]] = {
    "303(2)": (["cognizable", "non_cognizable"],
               "property_value_inr<5000=non_cognizable;else=cognizable"),
}


@dataclass
class Row:
    page: int
    cells: dict[str, list[str]] = field(default_factory=lambda: {c: [] for c in COLS})

    def text(self, col: str) -> str:
        return " ".join(self.cells[col]).strip()

    @property
    def section(self) -> str:
        # "58 (a)" is printed as two words; the Schedule means "58(a)"
        return "".join(self.cells["section"]).strip()


def _col_of(x0: float, cog_start: float, bail_start: float) -> str:
    if x0 < X_OFFENCE:
        return "section"
    if x0 < X_PUNISHMENT:
        return "offence"
    if x0 >= X_COURT:
        return "court"
    if x0 < cog_start - 2.0:
        return "punishment"
    if x0 < bail_start - 2.0:
        return "cognizable"
    return "bailable"


def _class_starts(ws: list[dict]) -> tuple[float | None, float | None]:
    """x0 of the cognizable and bailable cells on a row's first line, if the
    line carries them (cells are top-aligned, so the first line usually does)."""
    mid = [w for w in ws if X_MID_MIN <= w["x0"] < X_COURT]
    cog = next((w["x0"] for w in mid if w["text"].rstrip(".").lower() in COG_VOCAB), None)
    bail = None
    if cog is not None:
        bail = next((w["x0"] for w in mid
                     if w["x0"] >= cog + 50 and w["text"].rstrip(".").lower() in BAIL_VOCAB), None)
    return cog, bail


def _lines(words: list[dict]) -> list[tuple[float, list[dict]]]:
    """Group words into lines by `top`, each line sorted left to right."""
    words = sorted(words, key=lambda w: (w["top"], w["x0"]))
    lines: list[tuple[float, list[dict]]] = []
    for w in words:
        if lines and abs(w["top"] - lines[-1][0]) <= LINE_TOL:
            lines[-1][1].append(w)
        else:
            lines.append((w["top"], [w]))
    return [(t, sorted(ws, key=lambda w: w["x0"])) for t, ws in lines]


def _body_words(page) -> list[dict]:
    """Words below the '1 2 3 4 5 6' column-number line and above the footer."""
    words = page.extract_words(x_tolerance=1.5)
    digits = [w for w in words if w["text"] in "123456" and len(w["text"]) == 1]
    by_top: dict[int, set[str]] = defaultdict(set)
    for w in digits:
        by_top[round(w["top"])].add(w["text"])
    header_tops = [t for t, s in by_top.items() if s == set("123456")]
    if not header_tops:
        raise RuntimeError(f"page {page.page_number}: no column-number line found")
    cut = max(header_tops) + 5
    return [w for w in words if w["top"] > cut and w["top"] < page.height - 60]


def parse(pdf_path: Path) -> list[Row]:
    import pdfplumber

    rows: list[Row] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for pno in range(FIRST_PAGE, LAST_PAGE + 1):
            page = pdf.pages[pno - 1]
            prev_top: float | None = None
            page_first = True
            cog_start, bail_start = DEFAULT_COG_START, DEFAULT_BAIL_START
            for top, ws in _lines(_body_words(page)):
                sec_words = [w["text"] for w in ws if w["x0"] < X_OFFENCE]
                has_section = bool(sec_words) and _SECTION_RE.match("".join(sec_words)) is not None
                gap = (top - prev_top) if prev_top is not None else 0.0
                c0, b0 = _class_starts(ws)
                starts_class = c0 is not None
                new_row = has_section or (gap >= ROW_GAP and starts_class) or gap >= 14.5
                if page_first:
                    # a row may continue from the previous page: decide later
                    new_row = True
                    page_first = False
                if new_row or not rows:
                    rows.append(Row(page=pno))
                    # the classification cells of this row start where its
                    # first line says they do; keep the last known otherwise
                    if c0 is not None:
                        cog_start = c0
                    if b0 is not None:
                        bail_start = b0
                cols = {c: [w["text"] for w in ws if _col_of(w["x0"], cog_start, bail_start) == c]
                        for c in COLS}
                if not has_section and cols["section"]:
                    # stray text in column 1 that is not a section number
                    cols["offence"] = cols["section"] + cols["offence"]
                    cols["section"] = []
                for c in COLS:
                    rows[-1].cells[c].extend(cols[c])
                prev_top = top
    return _merge_continuations(rows)


def _merge_continuations(rows: list[Row]) -> list[Row]:
    """A page-top fragment with no section number and no classification is
    the tail of the previous row (the table broke across pages)."""
    out: list[Row] = []
    for r in rows:
        if out and not r.section and not r.text("cognizable") and not r.text("bailable"):
            for c in COLS:
                out[-1].cells[c].extend(r.cells[c])
        else:
            out.append(r)
    return out


def _classify(raw: str) -> str:
    t = raw.strip().rstrip(".").lower()
    if t == "cognizable":
        return "cognizable"
    if t == "non-cognizable":
        return "non_cognizable"
    if t:
        return "conditional"
    return "unknown"


def _bailable(raw: str) -> str:
    t = raw.strip().rstrip(".").lower()
    if t == "bailable":
        return "bailable"
    if t == "non-bailable":
        return "non_bailable"
    if t:
        return "conditional"
    return "unknown"


def write_rows(rows: list[Row], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["row", "gazette_page", "section", "offence", "punishment",
                    "cognizable_raw", "bailable_raw", "court"])
        for i, r in enumerate(rows, 1):
            w.writerow([i, r.page, r.section, r.text("offence"), r.text("punishment"),
                        r.text("cognizable"), r.text("bailable"), r.text("court")])


def build_table(rows: list[Row]) -> list[dict]:
    """One line per section key. Unlabelled variant rows inherit the last key."""
    groups: dict[str, list[Row]] = defaultdict(list)
    order: list[str] = []
    key = None
    for r in rows:
        if r.section:
            key = r.section
        if key is None:
            continue                       # text before the first numbered row
        if key not in groups:
            order.append(key)
        groups[key].append(r)

    table = []
    for key in order:
        grp = groups[key]
        # a caption row ("264 Omission to apprehend ... in case of --") carries
        # no classification of its own; its text prefixes the variants below it
        classified = [r for r in grp if r.text("cognizable") or r.text("bailable")] or grp
        caption = " ".join(r.text("offence") for r in grp if r not in classified)
        cogs = [_classify(r.text("cognizable")) for r in classified]
        bails = {_bailable(r.text("bailable")) for r in classified}
        notes: list[str] = []
        condition = ""
        if len(set(cogs)) == 1:
            cog = cogs[0]
            if cog == "conditional":
                notes.append(_clean(" | ".join(dict.fromkeys(r.text("cognizable") for r in classified))))
            elif cog == "unknown":
                notes.append("PARSE GAP: empty classification cell -- check bnss_schedule1_rows.csv")
        else:
            cog = "conditional"
            notes.append("Schedule splits this section: " + "; ".join(
                f"[{_clean(r.text('offence'))}] {_clean(r.text('cognizable'))}" for r in classified))
        if key in KNOWN_CONDITIONS:
            expected, condition = KNOWN_CONDITIONS[key]
            assert cogs == expected, f"{key}: Schedule rows now read {cogs}, KNOWN_CONDITIONS expects {expected}"
        bail = bails.pop() if len(bails) == 1 else ("conditional" if bails else "unknown")
        pages = sorted({r.page for r in grp})
        offences = [_clean(r.text("offence")) for r in classified]
        table.append({
            "bns_section": key,
            "offence": (caption + " " if caption else "") + (offences[0] if len(offences) == 1 else " / ".join(offences)),
            "punishment": " / ".join(_clean(r.text("punishment")) for r in classified),
            "cognizable": cog,
            "condition": condition,
            "bailable": bail,
            "triable_by": " / ".join(sorted({_clean(r.text("court")) for r in classified})),
            "source_ref": "BNSS 2023 First Schedule Pt I, Gazette of India Extraordinary No. 55 "
                          f"(25-12-2023) p. {'-'.join(str(p) for p in pages)}",
            "notes": "; ".join(notes),
        })
    return table


def _clean(text: str) -> str:
    """Undo line-wrap artefacts of the gazette typesetting ("non- cognizable")."""
    return re.sub(r"(\w)- (\w)", r"\1-\2", text).strip()


def write_table(table: list[dict], path: Path) -> None:
    cols = ["bns_section", "offence", "punishment", "cognizable", "condition", "bailable",
            "triable_by", "source_ref", "notes"]
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(table)


def check_against_stub(table: list[dict]) -> int:
    from fir.statute.cognizability import _STUB_SCHEDULE  # noqa: PLC2701 -- deliberate

    by_key = {t["bns_section"]: t for t in table}
    by_base: dict[str, set[str]] = defaultdict(set)
    for t in table:
        by_base[re.match(r"\d+", t["bns_section"]).group(0)].add(t["cognizable"])
    n_bad = 0
    print(f"\n{'BNS':>6}  {'stub':<15} {'schedule':<15} note")
    for sec, stub_val in _STUB_SCHEDULE.items():
        got = by_key.get(sec, {}).get("cognizable")
        if got is None:
            vals = by_base.get(sec)
            if not vals:
                got, note = "-- not in Schedule", ""
            elif len(vals) == 1:
                got, note = next(iter(vals)), "(all sub-clauses)"
            else:
                got, note = "mixed", "sub-clauses: " + ", ".join(sorted(vals))
        else:
            note = ""
        flag = "" if got == stub_val.value else "  <-- differs"
        if flag:
            n_bad += 1
        print(f"{sec:>6}  {stub_val.value:<15} {got:<15} {note}{flag}")
    print(f"\n{n_bad} stub entries differ from the parsed Schedule")
    return n_bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, default=PDF)
    ap.add_argument("--check", action="store_true", help="compare with the hand-coded stub")
    args = ap.parse_args()
    if not args.pdf.exists():
        print(f"missing {args.pdf}; download the MHA gazette PDF first (see docstring)")
        return 2

    rows = parse(args.pdf)
    write_rows(rows, OUT_ROWS)
    table = build_table(rows)
    write_table(table, OUT_TABLE)

    from collections import Counter
    dist = Counter(t["cognizable"] for t in table)
    print(f"{len(rows)} Schedule rows -> {len(table)} section keys "
          f"({dist['cognizable']} cognizable, {dist['non_cognizable']} non-cognizable, "
          f"{dist['conditional']} conditional, {dist['unknown']} unknown)")
    print(f"wrote {OUT_ROWS.relative_to(REPO)} and {OUT_TABLE.relative_to(REPO)}")
    if args.check:
        check_against_stub(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
