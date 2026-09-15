"""One place to load a statute classifier by name.

The graph, the server and the CLI all need "give me the classifier", and none
of them should know which artifact lives where or which model is currently
best. That decision belongs here and in `configs/pipeline.yaml`.

    tfidf   TfidfStatuteClassifier   -- the default. It currently beats the
                                        encoder 2:1 (PROGRESS.md finding 7/8).
    bert    InLegalBertStatuteClassifier
    auto    the config default, falling back to whichever artifact exists

Both models satisfy `fir.orchestrator.graph.StatuteClassifier`.
"""

from __future__ import annotations

from fir.config import ARTIFACT_DIR, pipeline

CHOICES = ("tfidf", "bert", "auto")

_ARTIFACTS = {
    "tfidf": ARTIFACT_DIR / "models" / "tfidf_statute.joblib",
    "bert": ARTIFACT_DIR / "models" / "inlegalbert_statute" / "statute_head.json",
}


def available() -> dict[str, bool]:
    """Which trained artifacts are on disk."""
    return {name: path.exists() for name, path in _ARTIFACTS.items()}


def default_name() -> str:
    """The configured default (`statute_id.default_classifier`), else tfidf."""
    return str(pipeline().get("statute_id", {}).get("default_classifier", "tfidf"))


def resolve(name: str = "auto") -> str:
    """Turn 'auto' into a concrete choice, preferring the config default."""
    if name != "auto":
        return name
    avail = available()
    pref = default_name()
    if avail.get(pref):
        return pref
    for candidate in ("tfidf", "bert"):
        if avail.get(candidate):
            return candidate
    return pref  # let load() raise the informative FileNotFoundError


def load(name: str = "auto"):
    """Load the named classifier. Raises FileNotFoundError with the command to
    run if its artifact is missing."""
    name = resolve(name)
    if name == "tfidf":
        from fir.statute.tfidf_baseline import TfidfStatuteClassifier

        return TfidfStatuteClassifier.load()
    if name == "bert":
        from fir.statute.inlegalbert import InLegalBertStatuteClassifier

        return InLegalBertStatuteClassifier.load()
    raise ValueError(f"unknown classifier {name!r}; choose from {CHOICES}")
