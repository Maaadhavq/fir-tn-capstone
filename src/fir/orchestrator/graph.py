"""LangGraph wiring for the first vertical slice.

    (audio) --> asr --+
                      |
    (text) -----------+--> extract --> translate --> statute_id --> ipc_bns --> cognizability
                                                                                     |
                                                                  END <-- instantiate

`translate` (Tamil -> English, IndicTrans2) produces `narrative_en` for the
classifier only; every other node keeps reading the original `narrative`, so
provenance offsets stay valid and Tamil cues stay first-class.

Stage A (ASR), the deterministic floor of Stage B (rule extraction of dates,
amounts, phones), Stage C (statute-ID + IPC->BNS + cognizability) and Stage D
(IF-1 record + rendered form). Learned extraction and the narrative composer are
not in yet; the `extract` node is where the LLM/NER extractor will slot in
alongside the rules. What the graph proves is that the stages compose, that
state flows between them, and that the officer-facing outputs are produced by
real code rather than by hand.

Every component is injected. The golden test drives the graph with a stub
classifier so it runs in milliseconds with no model weights on disk, exercising
the same node and edge logic the real run uses.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Annotated, Any, Protocol, TypedDict, runtime_checkable

from langgraph.graph import END, START, StateGraph

from fir.statute.cognizability import Route, route_case
from fir.statute.ipc_bns_map import IpcBnsMap


@runtime_checkable
class StatuteClassifier(Protocol):
    """What the graph needs from a statute-ID model.

    Both TfidfStatuteClassifier and InLegalBertStatuteClassifier satisfy this,
    as does the test stub.
    """

    name: str

    def predict_sections(self, text: str, threshold: float | None = ...) -> list[str]:
        ...

    def top_k(self, text: str, k: int = ...) -> list[tuple[str, float]]:
        ...


def _keep_last(_old: Any, new: Any) -> Any:
    return new


def _extend(old: list, new: list) -> list:
    return (old or []) + (new or [])


class SliceState(TypedDict, total=False):
    """State threaded through the graph."""

    # --- inputs ---
    audio_path: str
    narrative: str              # supplied directly, or filled in by the ASR node
    reference_transcript: str

    # --- stage A: ASR ---
    transcript: str
    asr_language: str
    asr_duration: float
    asr_device: str
    asr_flags: list[str]        # from fir.asr.quality; non-empty => not auto-resolvable

    # --- stage C: statute identification ---
    predicted_ipc: list[str]
    ipc_top_k: list[tuple[str, float]]
    classifier_name: str

    # --- IPC -> BNS ---
    bns_sections: list[str]
    auto_applied: list[dict]
    review_queue: list[dict]
    cue_hits: list[dict]        # sections whose ingredients the ORIGINAL text evidences but the
                                # classifier did not predict -- the recall safety net (see ipc_bns_node)

    # --- cognizability ---
    route: str
    cognizability: dict
    rationale: str

    # --- stage B: extraction (rules for now) ---
    extracted_spans: list[dict]

    # --- translation (classifier input only) ---
    narrative_en: str           # English text the classifier sees; == narrative when no translation
    translation: dict           # model, seconds, tamil_share -- or {} when skipped

    # --- stage D: instantiation ---
    fir_record: dict            # FirRecord.model_dump()
    if1_text: str               # rendered form, plain mode
    fill_report: dict
    narrative_grounded: bool    # the faithfulness gate

    # --- meta ---
    errors: Annotated[list[str], _extend]


# ---------------------------------------------------------------------------
# nodes
# ---------------------------------------------------------------------------


def make_asr_node(asr: Any):
    """Stage A: audio -> transcript."""

    def asr_node(state: SliceState) -> dict:
        path = state.get("audio_path")
        if not path:
            return {}
        if asr is None:
            return {"errors": ["asr_node: no ASR backend configured"]}
        try:
            tr = asr.transcribe(path)
        except Exception as exc:  # noqa: BLE001 -- surface, do not abort the graph
            return {"errors": [f"asr_node: {type(exc).__name__}: {exc}"]}
        return {
            "transcript": tr.text,
            "narrative": tr.text,
            "asr_language": tr.language,
            "asr_duration": tr.duration,
            "asr_device": getattr(asr, "resolved_device", ""),
            "asr_flags": list(getattr(tr, "flags", []) or []),
        }

    return asr_node


def make_translate_node(translator: Any):
    """Tamil -> English for the classifier. No-op (narrative_en = narrative)
    when the text is not Tamil-dominant, when no translator is configured, or
    when the model is not on disk -- the last case is reported in `errors`."""

    def translate_node(state: SliceState) -> dict:
        text = state.get("narrative") or ""
        base = {"narrative_en": text, "translation": {}}
        if translator is None or not text.strip():
            return base
        try:
            if not translator.should_translate(text):
                return base
            if not translator.available():
                return {**base, "errors": [
                    "translate_node: Tamil input but IndicTrans2 is not downloaded -- "
                    "classifier will see Tamil and likely predict nothing "
                    "(python scripts/download_indictrans.py)"
                ]}
            tr = translator.translate(text)
        except Exception as exc:  # noqa: BLE001
            return {**base, "errors": [f"translate_node: {type(exc).__name__}: {exc}"]}
        if tr is None:
            return base
        return {
            "narrative_en": tr.target,
            "translation": {"model": tr.model, "seconds": round(tr.seconds, 2),
                            "tamil_share": round(tr.tamil_share, 3), "sentences": tr.sentences},
        }

    return translate_node


def make_statute_node(classifier: StatuteClassifier, threshold: float | None = None):
    """Stage C: narrative -> predicted IPC sections."""

    def statute_node(state: SliceState) -> dict:
        # the classifier is English-only (ILSI); read the translated view when present
        text = (state.get("narrative_en") or state.get("narrative") or "").strip()
        if not text:
            return {
                "predicted_ipc": [],
                "errors": ["statute_node: empty narrative, nothing to classify"],
            }
        try:
            sections = classifier.predict_sections(text, threshold)
            top = classifier.top_k(text, 5)
        except Exception as exc:  # noqa: BLE001
            return {
                "predicted_ipc": [],
                "errors": [f"statute_node: {type(exc).__name__}: {exc}"],
            }
        return {
            "predicted_ipc": sections,
            "ipc_top_k": top,
            "classifier_name": getattr(classifier, "name", type(classifier).__name__),
        }

    return statute_node


def make_ipc_bns_node(ipc_map: IpcBnsMap):
    """IPC -> BNS, split into auto-applied and human-review -- plus the element
    cue scan as a recall safety net for the classifier.

    The classifier reads the English view; the cue scan reads the *original*.
    A covered section whose every ingredient is evidenced in the original but
    which the classifier did not predict is added to the review queue as
    `cues_present_not_predicted`. It is never auto-applied -- the cue lists are
    keyword heuristics, not a verdict -- but its presence in the queue stops a
    confident CSR/FIR route until an officer has looked (see route_case)."""
    from fir.statute.elements import scan_all
    from fir.statute.ipc_bns_map import ReviewReason

    def ipc_bns_node(state: SliceState) -> dict:
        from fir.statute.cognizability import classify_section

        predicted = state.get("predicted_ipc") or []
        result = ipc_map.apply(predicted)
        review = [{**asdict(item), "reason": item.reason.value} for item in result.review_queue]

        # Cue hits are a separate channel, not review items: they are questions
        # raised by keyword heuristics on the original text, and must not change
        # the output of a correct FIR route (a plain theft also trips 380's
        # cues -- a shop is a building). They only *force* review when they
        # would raise the route (see cognizability_node).
        already = set(predicted) | {r["ipc_section"] for r in review}
        ctx = _schedule_context(state)      # a stated value settles 303(2)'s Rs 5,000 split
        cue_hits = []
        for hit in scan_all(state.get("narrative") or ""):
            if hit.ipc_section in already:
                continue
            row = ipc_map.lookup(hit.ipc_section)
            bns = row.normalised_bns if (row and row.is_bns_section) else None
            cue_hits.append({
                "ipc_section": hit.ipc_section,
                "suggested_bns": bns,
                "offence": row.offence if row else "",
                "cognizable": classify_section(bns, ctx).value if bns else "unknown",
                "evidenced": hit.evidenced, "total": hit.total,
                "cues": list(hit.cues[:4]),
                "reason": ReviewReason.CUES_PRESENT_NOT_PREDICTED.value,
            })

        return {
            "bns_sections": result.bns_sections,
            "auto_applied": [asdict(r) for r in result.auto_applied],
            "review_queue": review,
            "cue_hits": cue_hits,
        }

    return ipc_bns_node


def _schedule_context(state: SliceState) -> dict:
    """Extracted facts a conditional Schedule entry may turn on.

    Mirrors fill.py: one unambiguous amount is the property value; several
    amounts are ambiguous and stay out, so the entry stays conditional and
    the officer is asked."""
    amounts = [sp for sp in (state.get("extracted_spans") or []) if sp.get("kind") == "amount_inr"]
    ctx: dict = {}
    if len(amounts) == 1 and amounts[0].get("value") is not None:
        ctx["property_value_inr"] = amounts[0]["value"]
    return ctx


def cognizability_node(state: SliceState) -> dict:
    """BNS sections -> FIR / CSR / OFFICER_REVIEW.

    The cue-scan safety net acts here: if nothing cognizable was auto-applied but
    the ORIGINAL narrative evidences every ingredient of a cognizable section the
    classifier missed, the case must not settle as CSR -- it goes to the officer
    with that section named. This is what stops a mistranslated "dowry" turning a
    498A complaint into a Community Service Register entry."""
    result = route_case(
        state.get("bns_sections") or [],
        has_review_items=bool(state.get("review_queue")),
        context=_schedule_context(state),
    )
    route, rationale = result.route.value, result.rationale
    if route != Route.FIR.value:
        # anything not settled as non-cognizable can raise the route: a
        # conditional or unknown hit could still be cognizable
        raising = [h for h in (state.get("cue_hits") or [])
                   if h.get("cognizable") != "non_cognizable"]
        if raising:
            names = ", ".join(f"IPC {h['ipc_section']} ({', '.join(h['cues'][:2])})" for h in raising)
            route = Route.OFFICER_REVIEW.value
            rationale = (f"original narrative evidences possibly cognizable offence(s) the classifier "
                         f"did not predict: {names}" + (f"; {result.rationale}" if result.rationale else ""))
    return {
        "route": route,
        "rationale": rationale,
        "cognizability": {s: c.value for s, c in result.per_section.items()},
    }


def extract_node(state: SliceState) -> dict:
    """Stage B floor: rule-extract structured slots from the narrative.

    Stores spans, not a record: the record is assembled once, in `instantiate`,
    after the statute decision exists.
    """
    from dataclasses import asdict as _asdict

    from fir.extract.rules import iter_spans

    text = state.get("narrative") or ""
    if not text.strip():
        return {"extracted_spans": []}
    try:
        spans = [{**_asdict(sp), "kind": sp.kind.value} for sp in iter_spans(text)]
    except Exception as exc:  # noqa: BLE001
        return {"extracted_spans": [], "errors": [f"extract_node: {type(exc).__name__}: {exc}"]}
    return {"extracted_spans": spans}


def instantiate_node(state: SliceState) -> dict:
    """Stage D: decision + extracted slots -> IF-1 record -> rendered form."""
    from fir.extract.fill import fill_structured_slots
    from fir.instantiate.from_decision import record_from_decision
    from fir.instantiate.narrative import attach_narrative
    from fir.instantiate.render import render_if1

    try:
        decision = to_decision(state)
        rec = record_from_decision(decision)
        rep = fill_structured_slots(rec, state.get("narrative") or "")
        attach_narrative(rec)   # after fill: the composer only writes from filled slots
        return {
            "fir_record": rec.model_dump(mode="json"),
            "if1_text": render_if1(rec),
            "fill_report": {
                "filled": sorted(rep.filled),
                "ambiguous": sorted(rep.ambiguous),
                "unplaced": [sp.value for sp in rep.unplaced],
            },
            "narrative_grounded": rec.narrative_is_grounded,
        }
    except Exception as exc:  # noqa: BLE001
        return {"errors": [f"instantiate_node: {type(exc).__name__}: {exc}"]}


# ---------------------------------------------------------------------------
# graph
# ---------------------------------------------------------------------------


def _entry(state: SliceState) -> str:
    """Skip the ASR node entirely for text input."""
    return "asr" if state.get("audio_path") else "extract"


def build_slice_graph(
    classifier: StatuteClassifier,
    asr: Any = None,
    ipc_map: IpcBnsMap | None = None,
    threshold: float | None = None,
    translator: Any = None,
):
    """Compile the slice graph.

    `asr` may be None when the graph will only ever see text input -- the ASR
    node is then unreachable via the conditional entry edge. `translator` may
    be None (no translation; Tamil input reaches the classifier untranslated).
    """
    ipc_map = ipc_map or IpcBnsMap.load()

    g = StateGraph(SliceState)
    g.add_node("asr", make_asr_node(asr))
    g.add_node("extract", extract_node)
    g.add_node("translate", make_translate_node(translator))
    g.add_node("statute_id", make_statute_node(classifier, threshold))
    g.add_node("ipc_bns", make_ipc_bns_node(ipc_map))
    g.add_node("cognizability_check", cognizability_node)
    g.add_node("instantiate", instantiate_node)

    g.add_conditional_edges(START, _entry, {"asr": "asr", "extract": "extract"})
    g.add_edge("asr", "extract")
    g.add_edge("extract", "translate")
    g.add_edge("translate", "statute_id")
    g.add_edge("statute_id", "ipc_bns")
    g.add_edge("ipc_bns", "cognizability_check")
    g.add_edge("cognizability_check", "instantiate")
    g.add_edge("instantiate", END)

    return g.compile()


def run_text(graph, narrative: str) -> SliceState:
    """Convenience: run the graph on a written complaint."""
    return graph.invoke({"narrative": narrative, "errors": []})


def run_audio(graph, audio_path: str, reference: str = "") -> SliceState:
    """Convenience: run the graph starting from an audio file."""
    return graph.invoke(
        {"audio_path": str(audio_path), "reference_transcript": reference, "errors": []}
    )


def to_decision(state: SliceState):
    """Validate a completed run into the officer-facing decision record.

    This is the boundary where loose graph state becomes the typed contract that
    Stage D (form instantiation) and Stage E (officer UI) consume.
    """
    from fir.schema.statute_decision import StatuteDecision

    return StatuteDecision.from_state(dict(state))


def format_result(state: SliceState) -> str:
    """Human-readable rendering of a graph run, used by the slice report."""
    lines: list[str] = []
    if state.get("transcript"):
        lines.append(f"  transcript   : {state['transcript'][:100]}")
        if state.get("asr_flags"):
            lines.append(f"  ! ASR flags  : {', '.join(state['asr_flags'])}")
    tr = state.get("translation") or {}
    if tr:
        lines.append(f"  translated   : ta->en via {tr.get('model', '?').split('/')[-1]} "
                     f"({tr.get('sentences', '?')} sent, {tr.get('seconds', '?')}s)")
        lines.append(f"  narrative_en : {(state.get('narrative_en') or '')[:100]}")
    lines.append(f"  IPC predicted: {', '.join(state.get('predicted_ipc') or []) or '(none)'}")
    lines.append(f"  BNS applied  : {', '.join(state.get('bns_sections') or []) or '(none)'}")

    review = state.get("review_queue") or []
    if review:
        detail = ", ".join(f"{r['ipc_section']}({r['reason']})" for r in review)
        lines.append(f"  to review    : {detail}")
    else:
        lines.append("  to review    : (none)")
    cues = state.get("cue_hits") or []
    if cues:
        detail = ", ".join(f"{h['ipc_section']}[{'/'.join(h['cues'][:2])}]" for h in cues)
        lines.append(f"  cues present : {detail}  (original text; not predicted)")

    route = state.get("route", "?")
    marker = {"FIR": "FIR", "CSR": "CSR", "OFFICER_REVIEW": "REVIEW"}.get(route, route)
    lines.append(f"  route        : {marker}  -- {state.get('rationale', '')}")

    fr = state.get("fill_report") or {}
    if fr:
        filled = ", ".join(fr.get("filled") or []) or "(none)"
        lines.append(f"  slots filled : {filled}")
        if fr.get("ambiguous"):
            lines.append(f"  ambiguous    : {', '.join(fr['ambiguous'])}")
    if "narrative_grounded" in state:
        gate = "PASS" if state["narrative_grounded"] else "FAIL"
        lines.append(f"  faithfulness : {gate}")

    for err in state.get("errors") or []:
        lines.append(f"  ! {err}")
    return "\n".join(lines)


__all__ = [
    "SliceState",
    "to_decision",
    "extract_node",
    "instantiate_node",
    "make_translate_node",
    "StatuteClassifier",
    "Route",
    "build_slice_graph",
    "run_text",
    "run_audio",
    "format_result",
]
