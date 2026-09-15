"""Statute text lookup and lexical retrieval over ILSI's 100 IPC sections."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from fir.config import DATA_DIR  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "ilsi" / "secs.jsonl").exists(), reason="ILSI secs.jsonl not downloaded"
)


@pytest.fixture(scope="module")
def index():
    from fir.statute.statute_text import StatuteIndex

    return StatuteIndex.load()


def test_index_covers_the_label_space(index):
    assert len(index) == 100
    assert index.text_for("302").startswith("Whoever commits murder")
    assert index.text_for("498A").startswith("Whoever, being the husband")
    assert index.text_for("99999") is None


def test_retrieval_finds_the_obvious_section(index):
    hits = index.search("the accused committed theft of a mobile phone", k=3)
    assert hits and hits[0].ipc_section in {"379", "378", "380"}   # theft cluster


def test_rank_of_corroborates_and_flags(index):
    text = "The victim was harassed for dowry by her husband and later found dead."
    assert index.rank_of(text, "304B") == 1            # dowry death: the words match
    assert index.rank_of(text, "498A") in (1, 2, 3)     # cruelty by husband
    assert index.rank_of(text, "34") is None            # common intention: no overlap


def test_decision_record_carries_statute_text_and_rank():
    from fir.orchestrator.graph import build_slice_graph, run_text, to_decision
    from fir.statute.ipc_bns_map import IpcBnsMap
    from tests.stubs import GOLDEN_RULES, StubStatuteClassifier

    g = build_slice_graph(classifier=StubStatuteClassifier(rules=GOLDEN_RULES), ipc_map=IpcBnsMap.load())
    d = to_decision(run_text(g, "The victim was harassed for dowry and later found dead."))
    by_ipc = {s.ipc_section: s for s in d.suggested_sections}
    assert by_ipc["304B"].statute_text.startswith("(1) Where the death of a woman")
    assert by_ipc["304B"].retrieval_rank == 1
    assert by_ipc["34"].retrieval_rank is None          # visible disagreement, not hidden

    d2 = to_decision(run_text(g, "the accused published defamatory statements"))
    assert d2.review_flags[0].statute_text.startswith("Whoever defames")


def test_punishment_boilerplate_does_not_drive_retrieval(index):
    """A complaint's 'fifty thousand rupees' must not retrieve sections by the size
    of their fine clause. Before the boilerplate stop list, this query returned
    448/342/323 -- all matching 'one thousand rupees' in the punishment text."""
    hits = index.search("someone stole my phone worth fifty thousand rupees", k=5)
    assert hits == [], [(h.ipc_section, round(h.score, 2)) for h in hits]


def test_inflection_is_normalised(index):
    """'cheated' must reach the section that says 'cheats'."""
    hits = index.search("he cheated me with a false promise", k=3)
    assert hits and hits[0].ipc_section in {"417", "420", "419"}
