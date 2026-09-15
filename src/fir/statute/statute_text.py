"""Statute text lookup and lexical retrieval over the IPC sections ILSI covers.

Two jobs, both small, both on the path to the element-wise verifier (PLAN P3):

1. **Lookup.** Given a predicted IPC section, return its statutory text so the
   officer sees the words of the law next to the model's suggestion, not just a
   number. Provenance for the *legal* side of the decision.

2. **Retrieval.** Given a complaint, rank the 100 section texts by lexical
   similarity. This is not a second classifier -- it is the retriever the
   element verifier will use to fetch candidate statute text for an LLM to
   check elements against, and until then it is a sanity signal: if the
   classifier says 302 but the retriever ranks 302 nowhere, that disagreement
   is worth showing.

TF-IDF cosine over ~100 short texts is the right tool here; BGE-m3 is planned
for the BNS/BNSS corpus once that is parsed, and would be overkill on 100
documents of ~300 characters.

BNS text is the eventual target. ILSI's `secs.jsonl` is IPC; the IPC->BNS map
gives the BNS number, and the BNS text will come from `data/statutes/` once it
exists (configs/datasets.yaml). The interface here does not change.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass

import numpy as np

from fir.data.ilsi import load_statute_texts, parse_ipc_section

#: Words of the punishment clause, present in nearly every section and carrying
#: no information about *which* offence. Also "whoever"/"shall"/"liable", the
#: statutory frame itself.
_PUNISHMENT_BOILERPLATE = frozenset({
    "whoever", "shall", "punished", "punishment", "imprisonment", "imprisoned", "fine",
    "fined", "liable", "extend", "term", "description", "either", "years", "year",
    "months", "month", "rupees", "thousand", "hundred", "life", "simple",
    "rigorous", "both", "also", "commits", "committed", "commit", "offence", "offences",
    # NOT "death": boilerplate in "punished with death" but content in 302/304B
    "section", "sections", "act", "code", "provided", "case", "cases", "person", "persons",
})


_TOKEN_RE = re.compile(r"[a-z][a-z'-]+")


def _stem(w: str) -> str:
    """Deliberately light suffix stripping -- enough to unify 'cheated/cheats/
    cheating' with 'cheat' and 'injuries' with 'injury' without a stemmer
    dependency. Over-stemming would merge unrelated legal terms."""
    if len(w) <= 4:
        return w
    for suf, repl in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)] + repl
    return w


def _stem_tokens(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower())]


@dataclass(slots=True)
class StatuteHit:
    ipc_section: str
    label: str
    text: str
    score: float


class StatuteIndex:
    """TF-IDF cosine retriever over the ILSI statute texts."""

    def __init__(self, texts: dict[str, str]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.labels = list(texts)
        self.sections = [parse_ipc_section(lab) or lab for lab in self.labels]
        self.texts = [texts[lab] for lab in self.labels]
        self._by_section = {s: i for i, s in enumerate(self.sections)}
        # legal text is short and formulaic; bigrams ("common intention",
        # "dowry death") carry most of the signal. The punishment clause every
        # section ends with ("...imprisonment which may extend to one year, or
        # with fine which may extend to one thousand rupees") is shared
        # boilerplate: left in, a complaint mentioning "fifty thousand rupees"
        # retrieves sections by the size of their *fine*. Strip it.
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        # stop words are matched against *tokenizer output*, i.e. stems -- so stem them
        stop = sorted({_stem(w) for w in ENGLISH_STOP_WORDS | _PUNISHMENT_BOILERPLATE})
        self._vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words=stop,
                                    tokenizer=_stem_tokens, token_pattern=None)
        self._X = self._vec.fit_transform(self.texts)

    @classmethod
    @functools.lru_cache(maxsize=1)
    def load(cls) -> "StatuteIndex":
        return cls(load_statute_texts())

    def __len__(self) -> int:
        return len(self.labels)

    # -- lookup -------------------------------------------------------------
    def text_for(self, ipc_section: str) -> str | None:
        i = self._by_section.get(ipc_section.strip().upper())
        return self.texts[i] if i is not None else None

    def hit_for(self, ipc_section: str) -> StatuteHit | None:
        i = self._by_section.get(ipc_section.strip().upper())
        if i is None:
            return None
        return StatuteHit(self.sections[i], self.labels[i], self.texts[i], 1.0)

    # -- retrieval ----------------------------------------------------------
    def search(self, query: str, k: int = 5) -> list[StatuteHit]:
        q = self._vec.transform([query])
        sims = (self._X @ q.T).toarray().ravel()
        order = np.argsort(-sims)[:k]
        return [
            StatuteHit(self.sections[i], self.labels[i], self.texts[i], float(sims[i]))
            for i in order
            if sims[i] > 0
        ]

    def rank_of(self, query: str, ipc_section: str) -> int | None:
        """1-based rank of `ipc_section` for `query`, or None if it scores 0.

        The agreement signal: a predicted section that the retriever also ranks
        highly is corroborated by the statute's own wording.
        """
        q = self._vec.transform([query])
        sims = (self._X @ q.T).toarray().ravel()
        i = self._by_section.get(ipc_section.strip().upper())
        if i is None or sims[i] <= 0:
            return None
        return int((sims > sims[i]).sum()) + 1
