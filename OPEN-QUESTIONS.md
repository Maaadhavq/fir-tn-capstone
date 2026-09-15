# OPEN-QUESTIONS.md — Clarifying Questions & Assumptions

> The brief is thorough but underspecified in places. Below: (A) resolved decisions, (B) numbered
> open clarifying questions grouped by area, (C) explicit assumptions made to let planning proceed.
> Per the brief, all of this is for human verification/amendment before code.

## A. Resolved (via clarifying Q&A)

- **R-1 GPU/VRAM:** 16 GB target; 12 GB documented as degraded fallback.
  ⚠️ **REOPENED 2026-08-20 by measurement — see Q-15.** The actual dev box has 8 GB.
- **R-2 Cloud posture:** strict local-only at every stage, including synthetic data generation.
- **R-3 Gold-set data:** synthetic-first; self-recorded human gold set staged later.
- **R-4 Output location:** `Downloads\fir-tn-capstone\`.

## B. Open clarifying questions

### Legal / domain
1. Is the exact TN IF-1 field list to be treated as authoritative from a specific official PDF you
   can provide, or should we reconstruct it from public FIR copies? (Field labels/order affect the
   schema + PDF template.)
2. Which special acts beyond MV Act, IT Act 2000, and TN Prohibition of Harassment of Woman Act must
   be in scope for statute ID at capstone quality vs. best-effort?
3. Should the CSR (non-cognizable) advisory path be fully built, or stubbed as a routing decision
   only, for this capstone?
4. Is a law-qualified reviewer available to validate gold-set section labels + cognizability? (Needed
   for R7/R11 mitigations and credible statute metrics.)
5. For "direction and distance from the police station" and "beat number" — is there station
   reference data (coordinates, beat maps) available, or are these officer-entered only?

### Data / evaluation
6. Confirm the proposed gold-set sizes (§5.3: 500 extraction / 5–10 h ASR / 100–150 E2E) vs a smaller
   MVP target — what annotation labor budget (person-hours) is realistic?
7. How many volunteer speakers, and what dialect/regional coverage, for the self-recorded set? (Tamil
   has substantial regional variation affecting ASR.)
8. Is there any (even small, anonymized) access to *real* complaint text/audio later, or is synthetic
   + self-recorded the ceiling for the whole project?
9. Should we target the LeSICiN comparison on the original IPC labels, on BNS-mapped labels, or both?
   (Affects how directly comparable our numbers are to published baselines.)

### Product / system
10. Guided-interview mode: in scope as a built feature this capstone, or design-only? (It adds a TTS
    + dialogue-state workstream.)
11. Speaker diarization: required, or optional/nice-to-have? (Affects P1.1 effort.)
12. Officer UI: is a functional prototype sufficient, or is usability testing with real officers
    expected/possible?
13. Single-station single-box, or must the design anticipate multi-station? (Affects DB/tenancy.)
14. Expected concurrency: one complaint at a time per box, or several? (Affects VRAM scheduling.)
15. Authentication/identity: is there an existing station identity system to integrate, or is a
    standalone RBAC sufficient?

### Compute (reopened by measurement)

15. **Which model generates the synthetic IF-1 data, given 8 GB of real VRAM?** (Blocks PLAN §5.2.)

    Measured 2026-09-10: RTX 5050 Laptop, 8,151 MiB total, **7,280 MiB free** with the display
    attached — call it ~7.1 GB usable. Qwen2.5-14B weights, before any KV cache:

    | quant | weights | fits 7.1 GB? |
    |---|---|---|
    | Q4_K_M (the planned config) | ~8.9 GB | no — ~25% of layers must go to CPU |
    | IQ4_XS | ~7.9 GB | no |
    | Q3_K_M | ~7.3 GB | marginal, leaves nothing for KV cache |
    | Q3_K_S | ~6.6 GB | yes, small context only |
    | **Qwen2.5-7B Q4_K_M** | **~4.7 GB** | **yes, comfortably** |

    Note 4-bit was always the plan — the ~9 GB figure in `configs/models.yaml` *is* Q4_K_M, not FP16.
    So quantisation alone does not rescue the 14B; the only way to fit it whole is 3-bit, and for
    *generating training data* that is likely the wrong trade — a Q3 14B may produce worse pairs than
    a Q4 7B, and every downstream stage inherits that quality.

    Rough throughput stakes for ~2,000 pairs at ~500 output tokens (~1M tokens):
    14B-Q4 with CPU offload at ~4–8 tok/s ≈ **35–70 h**; 7B-Q4 fully resident at ~25–40 tok/s ≈ **7–11 h**.

    **Action when next on mains power:** install Ollama, pull both `qwen2.5:14b-instruct-q4_K_M` and
    `qwen2.5:7b-instruct-q4_K_M` (~14 GB total), and measure real tok/s on an actual Tamil
    complaint→IF-1 prompt. **Benchmark plugged in** — laptop GPUs downclock substantially on battery,
    so DC-power numbers would understate both models and invalidate the comparison.

### Scope / process
16. Team size and whether workstreams can truly run in parallel (drives the §4 schedule).
17. Any institutional ethics-board (IRB) requirement for the human-voice gold set, and its lead time?
18. Priority if the schedule slips: protect statute research core (P3) vs full E2E polish — confirm
    P3 is the graded contribution to protect.

## C. Assumptions made (flag any to correct)

1. **Academic prototype**, not a live deployment; no actual filing/CCTNS integration (per "out of
   scope"). Evaluation is a study, not a field pilot.
2. **Fictitious PII only** throughout dev/eval; no real complainant data handled.
3. **Complaint length ~3 minutes** for the latency budget (§8); revisit if typical complaints are
   much longer.
4. **BNS 2023 / BNSS 2023 are the operative statutes** (in force 1 July 2024); IPC used only as a
   corpus source via the MHA mapping.
5. **English is the statute/verification-reasoning language**; Tamil is preserved for complainant
   spans and the bilingual narrative/form output.
6. **Small team or resourced solo**; schedule expressed as parallelizable workstreams.
7. **Two extraction approaches (A encoder-NER, B LLM) are both built and compared** (not one chosen
   upfront), per the brief.
8. **Stack:** Python, FastAPI, React, LangChain/LangGraph, PostgreSQL, Docker — as specified.
9. **Deliverable filenames** for the clarifying list = `OPEN-QUESTIONS.md` (brief did not name it).
10. **Synthetic labels come "for free"** from structured scenario seeds; only the human gold set needs
    manual annotation.
11. **One resident 8B LLM** serves extraction + verification + narrative to fit 16 GB; the 14B is
    offline-only for data generation.
