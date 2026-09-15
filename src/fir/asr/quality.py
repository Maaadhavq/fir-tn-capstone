"""Transcript quality guards.

Whisper fails in ways that WER does not distinguish but an FIR system must. On
the first 60-clip FLEURS-ta baseline, three clips scored WER > 100% for three
unrelated reasons:

  repetition loop   the decoder re-emitted the first 30s window a second time
                    (clip 1916, 45.9s -> 68 words for 23 spoken)
  language drift    mid-sentence the output slid into Cyrillic / Polish /
                    English garbage despite language="ta" (clip 1996)
  metric artefact   4 reference words vs 6 hypothesis words, CER 13% -- just an
                    agglutination boundary split. Not a fault at all.

The first two put words in a complainant's mouth. A noisy transcript degrades
statute identification; a *fabricated* one feeds invented facts into a document
an officer signs. So these are flagged here, and a flagged transcript is never
auto-resolvable downstream (see fir.schema.statute_decision).

Real Tamil complaints are code-switched, so Latin script is expected and is NOT
flagged. Only scripts with no business in a Tamil-English transcript are.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum


class TranscriptFlag(str, Enum):
    REPETITION_LOOP = "repetition_loop"
    FOREIGN_SCRIPT = "foreign_script"
    EMPTY = "empty"


#: Unicode scripts a Tamil / Tamil-English transcript may legitimately contain.
#: Everything alphabetic outside this set is a decoder drift, not speech.
_ALLOWED_SCRIPT_PREFIXES = ("TAMIL", "LATIN")


def _script_of(ch: str) -> str:
    """'TAMIL', 'LATIN', 'CYRILLIC', ... from the character's Unicode name."""
    try:
        return unicodedata.name(ch).split()[0]
    except ValueError:
        return "UNKNOWN"


def foreign_script_ratio(text: str) -> tuple[float, Counter]:
    """Fraction of alphabetic characters not in an allowed script, plus a
    per-script tally of the offenders for the report."""
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0, Counter()
    offenders: Counter = Counter()
    for c in alpha:
        s = _script_of(c)
        if not any(s.startswith(p) for p in _ALLOWED_SCRIPT_PREFIXES):
            offenders[s] += 1
    return sum(offenders.values()) / len(alpha), offenders


def duplicated_fraction(text: str, n: int = 4) -> tuple[float, int, str]:
    """Share of the transcript covered by word n-grams that occur more than once.

    Returns (fraction, max_repeat_count, most_repeated_ngram).

    Why a fraction rather than a max count: the real failure (clip 1916) was the
    decoder replaying the first 30s window *once*, so every 4-gram in that span
    appeared exactly twice -- a max-count threshold of 3 missed it entirely. But
    ~50% of the transcript's n-gram positions were duplicates, against ~0% for
    normal speech. Tamil repeats function words (மற்றும் = 'and') freely, but
    the surrounding words must match too for an n-gram to duplicate -- and on
    the real baseline no clean clip produced a single duplicated trigram.
    """
    w = text.split()
    if len(w) < n:
        return 0.0, 0, ""
    positions = [tuple(w[i : i + n]) for i in range(len(w) - n + 1)]
    counts = Counter(positions)
    dup_positions = sum(1 for g in positions if counts[g] > 1)
    gram, top = counts.most_common(1)[0]
    return dup_positions / len(positions), top, " ".join(gram)


@dataclass(slots=True)
class QualityReport:
    flags: list[TranscriptFlag] = field(default_factory=list)
    foreign_ratio: float = 0.0
    foreign_count: int = 0
    foreign_scripts: dict[str, int] = field(default_factory=dict)
    duplicated_fraction: float = 0.0
    max_ngram_repeat: int = 0
    repeated_phrase: str = ""

    @property
    def ok(self) -> bool:
        return not self.flags

    def summary(self) -> str:
        if self.ok:
            return "ok"
        parts = []
        if TranscriptFlag.REPETITION_LOOP in self.flags:
            parts.append(
                f"repetition: {self.duplicated_fraction * 100:.0f}% of transcript duplicated"
            )
        if TranscriptFlag.FOREIGN_SCRIPT in self.flags:
            scripts = ", ".join(f"{k}:{v}" for k, v in self.foreign_scripts.items())
            parts.append(f"foreign script: {self.foreign_count} chars ({scripts})")
        if TranscriptFlag.EMPTY in self.flags:
            parts.append("empty")
        return "; ".join(parts)


def assess(
    text: str,
    foreign_min_chars: int = 2,
    duplicate_threshold: float = 0.15,
    ngram: int = 3,
) -> QualityReport:
    """Run every guard on a transcript.

    Thresholds were set against the real 60-clip baseline, not guessed:

    * foreign script -- an absolute count, not a ratio. Clip 1996 had 7 Cyrillic
      + 1 Greek letters, which was only 5.2% of alphabetic characters because
      Tamil words are long; a ratio threshold was fragile. There is no
      legitimate reason for *any* Cyrillic in a Tamil-English complaint, so two
      such characters is enough to flag. One is tolerated as noise.

    * repetition -- fraction of duplicated n-gram positions. Measured on the
      60-clip baseline: clip 1916 (a single, slightly fuzzy replay) scores 41% at
      n=3 and 28% at n=4; **every other clip scores exactly 0%** at both. n=3 is
      used because Whisper's replays are not verbatim and trigrams survive the
      variation better. 15% leaves room for a complainant who genuinely repeats
      a short phrase (~7% on a 30-word statement) while catching any replay.

    Known gap: hallucinated *English* ("eddie devastated atises of super g" in
    clip 1996) passes, because Latin script is legitimate in code-switched
    speech. Catching that needs a plausibility model, not a character test.
    """
    rep = QualityReport()
    if not text or not text.strip():
        rep.flags.append(TranscriptFlag.EMPTY)
        return rep

    rep.foreign_ratio, offenders = foreign_script_ratio(text)
    rep.foreign_count = sum(offenders.values())
    rep.foreign_scripts = dict(offenders.most_common())
    if rep.foreign_count >= foreign_min_chars:
        rep.flags.append(TranscriptFlag.FOREIGN_SCRIPT)

    rep.duplicated_fraction, rep.max_ngram_repeat, rep.repeated_phrase = (
        duplicated_fraction(text, n=ngram)
    )
    if rep.duplicated_fraction >= duplicate_threshold:
        rep.flags.append(TranscriptFlag.REPETITION_LOOP)

    return rep
