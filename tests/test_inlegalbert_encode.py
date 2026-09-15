"""InLegalBERT input encoding -- tokenizer only, no model, no GPU."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def clf():
    from transformers import AutoTokenizer

    from fir.statute.inlegalbert import InLegalBertStatuteClassifier
    from fir.statute.labels import LabelSpace

    c = InLegalBertStatuteClassifier(label_space=LabelSpace.from_vocab(), max_seq_len=64)
    try:
        c.tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT", local_files_only=True)
    except Exception:  # noqa: BLE001
        pytest.skip("InLegalBERT tokenizer not in local cache")
    return c


LONG = " ".join(f"word{i}" for i in range(400))   # far more than 64 tokens
SHORT = "the accused stole a phone"


def test_head_truncation_shape_and_specials(clf):
    clf.truncation = "head"
    ids, mask = clf._encode([LONG, SHORT])
    assert ids.shape == (2, 64) and mask.shape == (2, 64)
    assert ids[0, 0].item() == clf.tokenizer.cls_token_id
    assert mask[0].sum().item() == 64            # long doc fills the budget
    assert mask[1].sum().item() < 64             # short doc is padded


def test_head_tail_keeps_both_ends(clf):
    clf.truncation = "head_tail"
    clf.tail_fraction = 0.25
    ids, mask = clf._encode([LONG])
    assert ids.shape == (1, 64)
    toks = clf.tokenizer.convert_ids_to_tokens(ids[0].tolist())
    assert toks[0] == "[CLS]" and toks[-1] == "[SEP]"
    body = [t for t in toks[1:-1] if t != "[PAD]"]
    assert len(body) == 62
    joined = clf.tokenizer.convert_tokens_to_string(body)
    # the head is the start of the document ("word0" tokenises as word + ##0) ...
    assert joined.startswith("word0 word1")
    # ... and the tail is the end of it: word399 must survive, which head-only drops
    assert "word399" in joined
    clf.truncation = "head"
    ids_head, _ = clf._encode([LONG])
    joined_head = clf.tokenizer.convert_tokens_to_string(
        clf.tokenizer.convert_ids_to_tokens(ids_head[0].tolist())
    )
    assert "word399" not in joined_head


def test_head_tail_short_doc_is_unchanged(clf):
    """Below the budget there is nothing to truncate; both modes must agree."""
    clf.truncation = "head"
    a, ma = clf._encode([SHORT])
    clf.truncation = "head_tail"
    b, mb = clf._encode([SHORT])
    assert a.tolist() == b.tolist() and ma.tolist() == mb.tolist()


def test_tail_fraction_governs_the_split(clf):
    clf.truncation = "head_tail"
    for frac in (0.0, 0.5):
        clf.tail_fraction = frac
        ids, _ = clf._encode([LONG])
        toks = clf.tokenizer.convert_ids_to_tokens(ids[0].tolist())
        body = [t for t in toks[1:-1] if t != "[PAD]"]
        joined = clf.tokenizer.convert_tokens_to_string(body)
        if frac == 0.0:
            assert "word399" not in joined       # no tail at all == head mode
        else:
            assert "word399" in joined
