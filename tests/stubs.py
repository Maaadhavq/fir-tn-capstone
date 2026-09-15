"""Deterministic stand-ins for trained models, used by the golden test.

The golden test pins the behaviour of the *pipeline* -- mapping policy, review
routing, FIR/CSR decision -- not the accuracy of a classifier. Substituting a
keyword stub for the real model keeps that test hermetic: it runs in
milliseconds, needs no GPU, and needs no weights on disk, so a regression in the
mapping policy is never masked by a model that failed to load.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fir.asr.whisper import Transcript


@dataclass(slots=True)
class StubStatuteClassifier:
    """Keyword -> IPC sections. No learning, fully deterministic."""

    rules: dict[str, list[str]] = field(default_factory=dict)
    name: str = "stub-classifier"

    def predict_sections(self, text: str, threshold: float | None = None) -> list[str]:
        lowered = text.lower()
        out: list[str] = []
        for keyword, sections in self.rules.items():
            if keyword in lowered:
                for s in sections:
                    if s not in out:
                        out.append(s)
        return out

    def top_k(self, text: str, k: int = 5) -> list[tuple[str, float]]:
        # Descending pseudo-scores so ordering assertions stay meaningful.
        secs = self.predict_sections(text)
        return [(s, round(1.0 - 0.1 * i, 3)) for i, s in enumerate(secs[:k])]


@dataclass(slots=True)
class StubAsr:
    """Returns a canned transcript per audio path."""

    transcripts: dict[str, str] = field(default_factory=dict)
    resolved_device: str = "stub"
    default: str = ""

    def transcribe(self, audio_path: str) -> Transcript:
        text = self.transcripts.get(str(audio_path), self.default)
        return Transcript(text=text, language="ta", language_probability=1.0,
                          duration=1.0)


#: Keyword rules covering every branch the golden fixture exercises.
GOLDEN_RULES: dict[str, list[str]] = {
    "dowry": ["302", "304B", "498A", "34"],
    "stole": ["379"],
    "theft": ["379"],
    "slapped": ["323", "506"],
    "defam": ["500"],
    "bribe": ["161"],
    "smashed": ["427"],
}


@dataclass(slots=True)
class StubTranslator:
    """Tamil -> English by lookup; anything Tamil-dominant not in the table
    becomes the `default`. Mirrors IndicTranslator's interface."""

    table: dict[str, str] = field(default_factory=dict)
    default: str = "the accused stole a mobile phone worth Rs 15,000"
    min_tamil_share: float = 0.3
    downloaded: bool = True
    resolved_device: str = "stub"

    def should_translate(self, text: str) -> bool:
        from fir.translate.indictrans import tamil_share

        return tamil_share(text) >= self.min_tamil_share

    def available(self) -> bool:
        return self.downloaded

    def translate(self, text: str, force: bool = False):
        from fir.translate.indictrans import Translation, tamil_share

        if not force and not self.should_translate(text):
            return None
        return Translation(source=text, target=self.table.get(text, self.default),
                           model="stub-indictrans", seconds=0.01,
                           tamil_share=tamil_share(text), sentences=1)
