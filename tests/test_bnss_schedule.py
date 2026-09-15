"""The BNSS 2023 First Schedule table (data/statutes/bnss_schedule1.csv).

Built by scripts/build_bnss_schedule.py from the gazette PDF. These tests pin
(a) the file's integrity, (b) a handful of classifications a lawyer can check
against the printed Schedule, and (c) the loader's derived-row and conditional
logic. They do not need the PDF or pdfplumber.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fir.statute import cognizability as cog  # noqa: E402
from fir.statute.cognizability import (  # noqa: E402
    Cognizability,
    _eval_condition,
    schedule,
)

CSV = REPO_ROOT / "data" / "statutes" / "bnss_schedule1.csv"
ROWS = REPO_ROOT / "data" / "statutes" / "bnss_schedule1_rows.csv"


@pytest.fixture(scope="module")
def table() -> list[dict]:
    with CSV.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# file integrity
# ---------------------------------------------------------------------------


def test_table_shape(table):
    assert len(table) == 438
    assert set(table[0]) >= {"bns_section", "offence", "punishment", "cognizable", "condition",
                             "bailable", "triable_by", "source_ref", "notes"}
    values = {t["cognizable"] for t in table}
    assert values == {"cognizable", "non_cognizable", "conditional"}, values
    assert not [t["bns_section"] for t in table if t["cognizable"] == "unknown"], "parse gap"


def test_every_row_cites_the_gazette(table):
    for t in table:
        assert re.search(r"Gazette of India Extraordinary No\. 55 \(25-12-2023\) p\. \d{3}", t["source_ref"]), t


def test_sections_are_in_schedule_order_and_unique(table):
    keys = [t["bns_section"] for t in table]
    assert len(keys) == len(set(keys))
    bases = [int(re.match(r"\d+", k).group(0)) for k in keys]
    assert bases == sorted(bases)
    assert bases[0] == 49 and bases[-1] == 356      # Part I runs from abetment to defamation


def test_conditional_rows_carry_the_schedules_words(table):
    for t in table:
        if t["cognizable"] == "conditional":
            assert t["notes"], t["bns_section"]
    by = {t["bns_section"]: t for t in table}
    assert by["303(2)"]["condition"] == "property_value_inr<5000=non_cognizable;else=cognizable"
    assert "5,000" in by["303(2)"]["offence"]
    assert by["85"]["notes"].startswith("Cognizable if information")
    assert by["49"]["notes"].startswith("According as offence abetted")


def test_rows_file_is_verbatim_and_paged():
    with ROWS.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 461
    assert all(158 <= int(r["gazette_page"]) <= 188 for r in rows)
    theft = [r for r in rows if r["section"] == "303(2)"]
    assert theft and theft[0]["cognizable_raw"] == "Cognizable."


# ---------------------------------------------------------------------------
# classifications a reviewer can check against the printed Schedule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("section,expected", [
    ("103(1)", "cognizable"),        # murder
    ("105", "cognizable"),           # culpable homicide
    ("64(1)", "cognizable"),         # rape
    ("74", "cognizable"),            # outraging modesty
    ("80(2)", "cognizable"),         # dowry death
    ("115(2)", "non_cognizable"),    # voluntarily causing hurt
    ("117(2)", "cognizable"),        # grievous hurt
    ("118(1)", "cognizable"),        # hurt by dangerous weapon
    ("126(2)", "cognizable"),        # wrongful restraint  -- stub had this wrong
    ("137(2)", "cognizable"),        # kidnapping
    ("223(a)", "cognizable"),        # disobedience to order -- stub had this wrong
    ("296", "cognizable"),           # obscene acts -- stub had this wrong
    ("309(4)", "cognizable"),        # robbery
    ("318(2)", "non_cognizable"),    # cheating (simple)
    ("318(4)", "cognizable"),        # cheating with delivery of property
    ("324(4)", "non_cognizable"),    # mischief 20k-1L
    ("324(5)", "cognizable"),        # mischief >= 1L
    ("329(3)", "cognizable"),        # criminal trespass -- stub had this wrong
    ("351(2)", "non_cognizable"),    # criminal intimidation
    ("351(3)", "non_cognizable"),
    ("352", "non_cognizable"),       # intentional insult
    ("356(2)", "non_cognizable"),    # defamation
    ("85", "conditional"),           # cruelty: depends on who reports
    ("303(2)", "conditional"),       # theft: depends on value
    ("62", "conditional"),           # attempt: follows the offence
])
def test_known_classifications(table, section, expected):
    by = {t["bns_section"]: t for t in table}
    assert by[section]["cognizable"] == expected, by[section]


# ---------------------------------------------------------------------------
# loader logic
# ---------------------------------------------------------------------------


def test_loader_derives_base_rows_only_where_sub_clauses_agree():
    s = schedule()
    assert not s.is_stub
    assert "103" in s.derived and s.table["103"] is Cognizability.COGNIZABLE
    assert "80" in s.derived and s.table["80"] is Cognizability.COGNIZABLE
    assert "324" in s.derived and s.table["324"] is Cognizability.UNKNOWN
    assert s.notes["324"].startswith("sub-clauses differ")
    assert s.source_refs["80"] == "derived from 80(2)"
    # the derived count is not reported as Schedule rows
    assert s.n_rows == 438 and len(s.table) > 438


def test_lookup_expands_composite_targets():
    s = schedule()
    assert s.lookup("BNS 351(2),(3)") is Cognizability.NON_COGNIZABLE   # parts agree
    assert s.lookup("BNS 324(4),(5)") is Cognizability.UNKNOWN           # parts disagree
    assert s.lookup("bns 103 (1)") is Cognizability.COGNIZABLE           # spacing/case
    assert s.lookup("BNS 999") is Cognizability.UNKNOWN


def test_note_for_joins_the_relevant_schedule_words():
    s = schedule()
    assert "person aggrieved" in s.note_for("BNS 85")
    assert "5,000" in s.note_for("BNS 303(2)")
    assert s.note_for("BNS 103(1)") == ""


@pytest.mark.parametrize("cond,ctx,expected", [
    ("property_value_inr<5000=non_cognizable;else=cognizable", {"property_value_inr": 4999}, Cognizability.NON_COGNIZABLE),
    ("property_value_inr<5000=non_cognizable;else=cognizable", {"property_value_inr": 5000}, Cognizability.COGNIZABLE),
    ("property_value_inr<5000=non_cognizable;else=cognizable", {"property_value_inr": "15000"}, Cognizability.COGNIZABLE),
    ("property_value_inr<5000=non_cognizable;else=cognizable", {}, None),                    # fact missing
    ("property_value_inr<5000=non_cognizable;else=cognizable", {"property_value_inr": "n/a"}, None),
    ("property_value_inr>=5000=cognizable", {"property_value_inr": 100}, None),              # no else, no hit
    ("garbage", {"property_value_inr": 100}, None),
])
def test_eval_condition(cond, ctx, expected):
    assert _eval_condition(cond, ctx) is expected


def test_conditional_stays_conditional_without_the_fact(monkeypatch, tmp_path):
    csv_dir = tmp_path / "statutes"
    csv_dir.mkdir()
    (csv_dir / "bnss_schedule1.csv").write_text(
        "bns_section,cognizable,condition,notes,source_ref\n"
        "303(2),conditional,property_value_inr<5000=non_cognizable;else=cognizable,value split,p.181\n"
        "85,conditional,,who reports,p.162\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cog, "DATA_DIR", tmp_path)
    cog.schedule.cache_clear()
    try:
        s = cog.schedule()
        assert s.lookup("BNS 303(2)") is Cognizability.CONDITIONAL
        assert s.lookup("BNS 303(2)", {"property_value_inr": 200}) is Cognizability.NON_COGNIZABLE
        assert s.lookup("BNS 85", {"property_value_inr": 200}) is Cognizability.CONDITIONAL
        # derived base inherits a single sub-clause's condition
        assert s.lookup("BNS 303", {"property_value_inr": 200000}) is Cognizability.COGNIZABLE
        r = cog.route_case(["BNS 85"])
        assert r.route is cog.Route.OFFICER_REVIEW
        assert "who reports" in r.rationale and r.conditional_sections == ["BNS 85"]
    finally:
        cog.schedule.cache_clear()
