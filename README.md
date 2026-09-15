# Speech-Driven FIR Drafting System (TN IF-1)

Turns a spoken Tamil / Tamil-English citizen complaint into a **draft** First
Information Report in the Tamil Nadu Police CCTNS IF-1 format.

**This is decision support.** It drafts for mandatory officer verification. The
officer is the author of record; the system never files anything.

## Pipeline

```
A. Speech ingestion & ASR
B. Information extraction (NLU)
C. Statute identification          <- research core
D. Deterministic FIR instantiation
E. Officer verification UI
```

The IF-1 form is fixed and standard. We do not design or generate the document —
Stage D fills the fixed template by deterministic slot-filling. No AI writes into
structured fields. The one free-text box is auto-assembled from extracted facts
and must pass a faithfulness gate before the officer edits it.

## Status — Stage A/B-floor/C/D backbone runs end to end (2026-09-13)

What runs today, as one LangGraph graph:

```
(audio) → asr → extract → statute_id → ipc_bns → cognizability → instantiate → IF-1 draft
(text)  ────────┘
```

| stage | what is built | what is not |
|---|---|---|
| **A** ASR | faster-whisper large-v3 int8; repetition-loop and foreign-script guards calibrated on real output; flagged transcripts are never auto-resolvable; Tamil+English ITN (spoken numbers with sandhi → digits, offset-mapped); Ta→En translation node for the classifier (interim opus-mt, chrF 35) with the element cue scan on the original text as a recall safety net | IndicWhisper benchmark, code-switch LoRA, a better translator (IndicTrans2 needs your HF login) |
| **B** extraction | rule-based floor with provenance: dates (incl. spoken, year-less), times, INR amounts, phones, plates; digit and spoken forms alike; fills a slot only when unambiguous; 63/63 on the 28-doc gold smoke set, zero FP | NER (MuRIL), LLM structured extraction, synthetic training data |
| **C** statute-ID | TF-IDF + OvR logistic and InLegalBERT head on ILSI (66k docs, 100 IPC sections); IPC→BNS map with high-confidence-only auto-apply; cognizability routing from the parsed BNSS First Schedule (438 keys, gazette-cited, conditional entries ask the officer); BNS 2023 section text (gazette-parsed, 358 sections) shown on every suggestion with the IPC text as lineage, plus retrieval-rank corroboration; **element-wise justification v0** (29 station-house sections, EN+TA cues, yes-with-provenance or unclear, never no; doubles as the cue-scan recall net) | LLM element verifier, RAG over BNS text, legal read-through of the Schedule CSV |
| **D** instantiation | Pydantic IF-1 record in parity with the canonical schema; deterministic 15-item form renderer; template narrative composer whose sentences are grounded by construction | LLM narrative |
| **E** officer UI | FastAPI surface returning decision + record + rendered form | React verification screen |

Numbers so far (see `PROGRESS.md` for the honest caveats), all on the same 13,039 ILSI test docs
after training on all 42,835:

| model | micro-F1 | macro-F1 |
|---|---|---|
| TF-IDF + OvR logistic | **0.7000** | **0.5805** |
| InLegalBERT (2 epochs, pos_weight) | 0.4636 | 0.3984 |
| InLegalBERT (4 ep, head LR ×10, head+tail) | 0.5256 | 0.4594 |
| ensemble, best α on dev | 0.7006 | 0.5839 |

The cheap sparse baseline beats the domain-pretrained encoder with every lever pulled, and the
ensemble adds nothing — the encoder is redundant with the n-grams, not complementary. **TF-IDF is the
statute classifier; the encoder's job is element verification.** Six runs, ~3.5 GPU-hours, all in
`artifacts/reports/RESULTS.md`.
Whisper large-v3 zero-shot on FLEURS-ta: **WER 51.41% / CER 15.05%** over 60 clips.

278 tests, ~5s, no model weights needed.

## Quick start

Everything runs inside WSL2 Ubuntu — Smart App Control blocks `pyarrow` on the
Windows host, and the HF training stack depends on it. See `environment/README.md`.

```bash
bash environment/setup_wsl.sh   # Python 3.11 via uv, torch cu128, deps
make data                       # ILSI (512 MB) + FLEURS-ta test (407 MB)
make test                       # 368 tests, ~20s, no weights needed
make lint                       # ruff
make schedule                   # rebuild the BNSS First Schedule CSV from the gazette PDF
make bns-text                   # rebuild the 358 BNS section texts from the gazette PDF
make slice                      # F1 table + WER, subsampled
make report                     # re-print results + a rendered IF-1, no retraining
make serve                      # FastAPI on 127.0.0.1:8000
```

No `make` in your WSL? `bash scripts/run.sh <target>` runs the same commands.

## Two invariants you should not "fix"

**1. The IPC→BNS auto-apply rule.** Our training data (ILSI) is IPC-labelled;
police have filed under the BNS since 1 July 2024. The mapping table was built
from a third-party pocket directory, **not** the official MHA gazette, so only
rows marked `confidence == high AND needs_review == N` are applied without a
human. 29 of 100 rows are flagged, and rows targeting `OMITTED`, `VERIFY`, or the
Prevention of Corruption Act are routed out regardless of confidence — putting a
non-existent BNS section on an FIR is a legal error, not a metrics regression.

**2. Cognizability never defaults.** Cognizable → FIR, non-cognizable → CSR, and
anything unclassified → the officer. Refusing an FIR on a cognizable offence is
itself misconduct, so an unknown section must never silently become a CSR.

Both are pinned by `tests/test_slice_golden.py`. If a change breaks those tests,
the change is wrong unless the legal position actually changed.

## Layout

```
environment/   WSL setup, pinned requirements
configs/       datasets.yaml · models.yaml · pipeline.yaml
data/          gitignored, except data/mapping/ (our own artifact)
src/fir/
  schema/      if1 (canonical IF-1 record) · statute_decision · if1_v1.schema.json
  data/        ILSI + FLEURS loaders, no pyarrow anywhere
  asr/         whisper wrapper · quality guards · itn (spoken numbers)
  extract/     rules (dates/times/amounts/phones/plates) · fill
  statute/     labels · tfidf_baseline · inlegalbert · registry · ipc_bns_map · cognizability · statute_text · elements
  instantiate/ from_decision · narrative · render
  orchestrator/LangGraph wiring
  serving/     FastAPI app
harness/       metrics + runnable evaluations
tests/         golden regression
```

## Documents

`CONTEXT.md` is the single source of truth. `PROGRESS.md` is the session handoff —
read it first. `PLAN.md` (roadmap), `ARCHITECTURE.md`, `IMPLEMENTATION.md`
(engineering plan), `DATASETS.md` (verified data), `OPEN-QUESTIONS.md`.

## Legal basis

BNS 2023 & BNSS 2023, in force 1 July 2024. Cognizability per the BNSS First
Schedule. IPC-era corpora mapped via the MHA IPC↔BNS correspondence tables.
