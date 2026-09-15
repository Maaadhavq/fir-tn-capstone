"""BNS 2023 section text (data/statutes/bns_2023.jsonl) and its read side."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from fir.statute.bns_text import BnsText, bns_text  # noqa: E402

JSONL = REPO_ROOT / "data" / "statutes" / "bns_2023.jsonl"


@pytest.fixture(scope="module")
def records() -> list[dict]:
    with JSONL.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


# ---------------------------------------------------------------------------
# the file
# ---------------------------------------------------------------------------


def test_all_358_sections_in_order_with_headings(records):
    assert [r["section"] for r in records] == [str(i) for i in range(1, 359)]
    assert all(r["heading"] for r in records)
    assert all(r["text"] and r["operative_text"] for r in records)
    assert len({r["chapter"] for r in records}) == 20
    assert all(1 <= r["gazette_page"] <= 102 for r in records)


@pytest.mark.parametrize("section,heading,chapter", [
    ("1", "Short title, commencement and application.", "I"),
    ("2", "Definitions.", "I"),
    ("80", "Dowry death.", "V"),
    ("85", "Husband or relative of husband of a woman subjecting her to cruelty.", "V"),
    ("101", "Murder.", "VI"),
    ("103", "Punishment for murder.", "VI"),
    ("303", "Theft.", "XVII"),
    ("318", "Cheating.", "XVII"),
    ("351", "Criminal intimidation.", "XIX"),
    ("356", "Defamation.", "XIX"),
    ("358", "Repeal and savings.", "XX"),
])
def test_known_headings(records, section, heading, chapter):
    r = next(r for r in records if r["section"] == section)
    assert r["heading"] == heading
    assert r["chapter"] == chapter


def test_marginal_act_citations_are_not_headings(records):
    """The gazette prints '45 of 1860.' etc. in the margin beside references
    to other Acts; those are citations, not section headings."""
    import re

    for r in records:
        assert not re.search(r"\d+ of \d{4}", r["heading"]), (r["section"], r["heading"])


def test_no_running_heads_or_signature_block_in_text(records):
    for r in records:
        assert "GAZETTE OF INDIA" not in r["text"], r["section"]
    last = records[-1]["text"]
    assert "Legislative Counsel" not in last and "GOVERNMENT OF INDIA PRESS" not in last
    assert last.rstrip("— ").endswith("effect of the repeal.")


def test_operative_text_drops_illustrations_but_keeps_sub_sections(records):
    murder = next(r for r in records if r["section"] == "101")
    assert "A shoots Z" in murder["text"]
    assert "A shoots Z" not in murder["operative_text"]
    assert "Exception 1." in murder["operative_text"]           # exceptions are law, kept
    theft = next(r for r in records if r["section"] == "303")
    assert "(2) Whoever commits theft shall be punished" in theft["operative_text"]
    assert "five thousand rupees" in theft["operative_text"]     # the Rs 5,000 proviso


# ---------------------------------------------------------------------------
# the read side
# ---------------------------------------------------------------------------


def test_loader_and_lookup():
    bt = bns_text()
    assert bt is not None and len(bt) == 358
    assert bt.heading_for("BNS 103(1)") == "Punishment for murder."
    assert bt.section("BNS 999") is None and bt.text_for("BNS 999") is None
    assert bt.text_for("not a section") is None


def test_clause_lookup_returns_the_named_sub_section():
    bt = bns_text()
    t = bt.text_for("BNS 303(2)")
    assert t.startswith("(2) Whoever commits theft shall be punished")
    assert "is said to commit theft" not in t                   # (1) is the definition, not asked for
    assert bt.text_for("BNS 303").startswith("(1) Whoever, intending to take dishonestly")
    # a clause the gazette does not print as a numbered paragraph falls back to the section
    assert bt.text_for("BNS 74") == bt.section("74").operative_text


def test_composite_target_joins_each_clause():
    bt = bns_text()
    t = bt.text_for("BNS 324(4),(5)")
    assert "(4)" in t and "(5)" in t
    assert "twenty thousand rupees" in t and "one lakh rupees" in t


def test_missing_file_means_none_not_crash(tmp_path, monkeypatch):
    from fir.statute import bns_text as mod

    monkeypatch.setattr(mod, "DATA_DIR", tmp_path)
    BnsText.load.cache_clear()
    try:
        assert bns_text() is None
    finally:
        BnsText.load.cache_clear()


def test_decision_record_carries_the_bns_words():
    from fir.orchestrator.graph import build_slice_graph, run_text, to_decision
    from fir.statute.ipc_bns_map import IpcBnsMap
    from tests.stubs import GOLDEN_RULES, StubStatuteClassifier

    g = build_slice_graph(classifier=StubStatuteClassifier(rules=GOLDEN_RULES), ipc_map=IpcBnsMap.load())
    d = to_decision(run_text(g, "the accused stole a mobile phone worth Rs 15,000"))
    s = next(s for s in d.suggested_sections if s.bns_section == "BNS 303(2)")
    assert s.bns_heading == "Theft."
    assert s.bns_text.startswith("(2) Whoever commits theft")
    assert s.statute_text.startswith("Whoever")                 # the IPC lineage text is still there
    d2 = to_decision(run_text(g, "published defamatory statements"))
    f = next(f for f in d2.review_flags if f.ipc_section == "500")
    assert f.bns_heading == "Defamation." and f.bns_text
