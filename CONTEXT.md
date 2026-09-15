# Project Context — Speech-Driven FIR Drafting System (TN IF-1)

> Single source of truth for this capstone. Shared across Claude Code, Claude chat, and Cowork.
> Edit THIS file; then mirror it (see "Keeping surfaces in sync" at the bottom).

## What this is
A two-semester capstone: an end-to-end system that turns a spoken Tamil / Tamil-English citizen
complaint into a draft First Information Report in the Tamil Nadu Police format (CCTNS IF-1).
It is decision support — it drafts for mandatory officer verification; it never files.

## Pipeline (five stages)
A. Speech ingestion & ASR → B. Information extraction (NLU) → C. Statute identification (research core)
→ D. Deterministic FIR instantiation → E. Officer verification UI.

**SCOPE — FIR form is FIXED.** The Tamil Nadu FIR (IF-1) is a fixed, standard form. We do NOT design
or generate the document — Stage D just fills the fixed template with extracted details (deterministic
slot-filling; no AI writes into structured fields, no hallucinated structure). The only free-text is
the "what happened" narrative box, which is **auto-assembled (decided)** as a grounded factual summary
of the extracted facts — every sentence must trace to an extracted fact and pass the faithfulness gate
(no invented content), then the officer edits it. Kept minimal. So "generation" ≈ form-filling; the
real work is Stages B and C/D (pulling out details + choosing the right BNS sections).

## Locked decisions
- Compute: single 16 GB consumer GPU (12 GB = degraded fallback).
  ⚠️ **The actual dev machine has 8 GB (RTX 5050 Laptop), below the stated fallback.** The vertical
  slice fits; Qwen2.5-14B 4-bit for synthetic data-gen does not. This decision is *unchanged and still
  open* — see PROGRESS.md "Findings that change assumptions" #1 and `configs/models.yaml`, where each
  model carries a `fits: true|false` flag against the real budget.
- Cloud posture: strict local-only, always — even synthetic-data generation uses a local LLM.
- Eval data: synthetic-first; self-recorded human gold set staged later.
- Docs location: Downloads\fir-tn-capstone\.

## Stack & constraints
Python · FastAPI · React · LangChain/LangGraph · PostgreSQL · Docker. PEFT (LoRA/QLoRA) + quantized
inference. On-prem, offline-degradable. PII minimization, DPDP Act 2023 alignment.

## Key model choices
ASR: Whisper large-v3 (int8) / IndicWhisper / IndicConformer (12 GB fallback).
Resident LLM (extraction + verify + narrative): Qwen2.5-7B 4-bit.
Synthetic data-gen (offline, GPU-alone): Qwen2.5-14B 4-bit.
NER: MuRIL. Statute classifier: InLegalBERT. Retriever: BGE-m3.

## Legal basis
BNS 2023 & BNSS 2023 (in force 1 Jul 2024). IPC-era corpora (ILSI/LeSICiN) mapped via the official
MHA IPC↔BNS tables. Cognizability via BNSS First Schedule → cognizable = FIR, non-cognizable = CSR.

## Status
**Implementation — Stage A/B-floor/C/D backbone runs end to end (2026-09-13; Madhav building solo).**
One LangGraph graph takes a transcript or audio through: ASR with hallucination guards and Tamil ITN →
rule-based extraction with provenance → statute-ID (TF-IDF and InLegalBERT on all 66k ILSI docs) →
IPC→BNS under the high-confidence-only rule → cognizability from the parsed BNSS First Schedule (438
keys, gazette-cited; conditional entries ask the officer) → element-wise justification (v0), BNS section text from the gazette beside every suggestion → IF-1
record → grounded narrative → the fixed 15-item form. FastAPI + a verification page serve it. A Tamil→English
translation node (interim opus-mt model; IndicTrans2 is gated) feeds the English-only classifier.
Not built: learned extraction, the LLM narrative/verifier, and the React UI. Planning docs and Review-2 deliverables remain complete.
**PROGRESS.md is the live status — read it first**; it records the 8 GB hardware finding that contradicts
the compute decision below, and the full experiment log.

## Glossary
FIR = First Information Report · IF-1 = CCTNS Integrated Investigation Form 1 · BNS/BNSS = Bharatiya
Nyaya Sanhita / Nagarik Suraksha Sanhita 2023 · CCTNS = Crime & Criminal Tracking Network & Systems ·
CSR = Community Service Register · cognizable = police may act/investigate without prior court warrant.

## Keeping surfaces in sync (this is the "bridge")
- Claude Code: `CLAUDE.md` in this folder imports this file (`@CONTEXT.md`) — auto-loaded when working here.
- Chat / Cowork: create a Claude.ai Project, paste this file into Project knowledge (and/or custom
  instructions). Cowork sessions launched inside that project inherit it.
- When this file changes: re-paste into the Project, or connect this folder via an MCP/Drive connector
  for automatic sync.
