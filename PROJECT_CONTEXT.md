# PROJECT_CONTEXT.md — everything a teammate (or an LLM) needs to present this project

> Written 15 Sep 2026, the evening before Review 3. If you are making slides, a report, a poster or
> answering questions about this project, **start here and treat this file as authoritative.** Every
> number below is copied from machine-written reports (`artifacts/reports/RESULTS.md`); if a number
> is not in this file or in RESULTS.md, it does not exist — do not estimate one.
> The short "source of truth" the code assistant loads is `CONTEXT.md`; the dated engineering log with
> all findings is `PROGRESS.md`; this file is the long-form handoff that joins them.

---

## 0. If you are an LLM building a deck from this repo

- Use only numbers from §5 (identical to `artifacts/reports/RESULTS.md`). Quote them with their
  caveats (§6). Never round 51.41% to "about 50%" without saying so; never invent a baseline.
- The system is **decision support**: it *drafts* an FIR for an officer to verify. It never files, never
  decides guilt, never "detects crimes". Say "suggests sections", "drafts", "flags for the officer".
- The existing deck `Review3_Presentation.pptx` (18 slides) and its generator
  `scripts/review3/build_deck.py` are the reference storyline; `assets/r3_*.png` are the figures.
  Palette: navy `#24405B`, teal `#35637E`, grey `#5B6470`, panel `#F5F7F9`, red `#B4423B`, amber
  `#C98A1B`, green `#3C8A5A`; font Calibri; 16:9.
- Findings are numbered (§7). Cite them as "finding 6f" etc.; they map to sections of `PROGRESS.md`.
- Three things only the humans can supply: the panel's Review 2 comments and the action taken;
  each member's actual contribution; the AI-tool acknowledgement wording (§11). Leave visible
  placeholders `«…»` rather than guessing.

---

## 1. The project in one paragraph

**Speech-Driven Drafting of First Information Reports for the Tamil Nadu Police (CCTNS IF-1)** —
a B.Tech capstone (BCSE497J Project-I, VIT Chennai, SCOPE, AY 2026-27). A citizen speaks a
complaint in Tamil or Tamil-English; the system transcribes it, extracts the facts the fixed IF-1
form needs (dates, times, amounts, phone numbers, vehicle plates), identifies the applicable
Bharatiya Nyaya Sanhita (BNS) 2023 sections with element-wise justification, decides whether the
matter is cognizable (FIR) or not (CSR advisory) from the BNSS First Schedule, fills the fixed
15-item form deterministically, composes a grounded narrative, and presents a **draft** for the
officer to verify. Everything runs locally on one laptop GPU; nothing is filed automatically.

- Team: **Madhav K (23BAI1088), Nischay Kuchibotla (23BAI1245), Rohit A. S. (23BAI1416)**.
  Guide: **Dr. Shivaranjani**. Course: BCSE497J Project-I, Fall 2026-27.
- Review dates: Review 1 (9–11 Jul), Review 2 panel (19 Aug, done), **Review 3 panel 16 Sep 2026**,
  Review 4 guide (12–16 Oct), Draft report review (12–16 Oct), Review 5 final panel (21 Oct).
- Review 3 rubric (20 marks): follow-up on Review 2 feedback + progress vs plan (3); implementation
  ≈50% of scope (3); technical accuracy (3); interim results with metrics/graphs (3); problem-solving
  and refinement (2); report progress Ch 1–3 + initial Ch 4 (3); presentation, individual
  responsibility, Q&A (3). Source: `C:\Users\madha\Downloads\BCSE497J_Project_I_Guidelines.pdf`.

---

## 2. Hard facts that must not be misstated

| Fact | Detail |
|---|---|
| Decision support only | The officer is the author of record. No endpoint, CLI or graph node can set a record to `verified`; a test fails if one appears. The form prints "DRAFT". |
| Strict local-only | All models load from a local Hugging Face cache (`HF_HUB_OFFLINE=1`); the API binds 127.0.0.1; uploaded audio is deleted after transcription; no hosted API of any kind, including for synthetic data. |
| The IF-1 form is fixed | We do not design a document. Stage D is deterministic slot-filling into the standard TN CCTNS IF-1 (15 items). The only free text is the narrative box, auto-assembled from extracted facts and gated: every sentence must trace to a fact. |
| Legal basis | BNS 2023 and BNSS 2023 (in force 1 Jul 2024). Training data (ILSI) is IPC-era; an IPC→BNS map bridges it. Cognizability from the BNSS First Schedule. |
| Hardware truth | The dev machine has an **RTX 5050 Laptop, 8 GB VRAM** (Blackwell), WSL2 with 7.7 GB RAM. The plan assumed 16 GB (12 GB fallback). The pipeline fits; a 14B 4-bit LLM does not (finding 1). |
| Safety invariants | IPC→BNS auto-apply only when `confidence == high AND needs_review == N AND target is a BNS section`; cognizability never defaults (unknown → officer); a flagged ASR transcript blocks auto-resolution. |
| Who built it | The implementation from 20 Aug to 15 Sep was done by Madhav with an AI coding assistant (Claude Code); the guidelines require significant AI use to be acknowledged. See §11 before writing the contribution slide. |

---

## 3. What runs today (Review 3 state) — one LangGraph graph

```
(audio → asr) → extract → translate → statute_id → ipc_bns → cognizability_check → instantiate → END
```

| Stage | What is built | Status | Module |
|---|---|---|---|
| **A  Speech / ASR** | faster-whisper large-v3 (CTranslate2 int8) with device fallback; one decoding pass at temperature 0; batched VAD-chunk decoding; transcript guards for replay loops and foreign-script drift; Tamil inverse text normalisation (206-form number lexicon with sandhi, spoken dates/times/amounts → digits, offset map back to the audio) | working | `src/fir/asr/{whisper,quality,itn}.py` |
| **B  Extraction** | rule-based extractor with character-offset provenance: dates (digit and spoken, year-less), times, ₹ amounts (incl. lakh/crore words, Tamil numerals), phones, vehicle plates; fills a slot only when unambiguous | working floor (NER / LLM extraction not started) | `src/fir/extract/{rules,fill}.py` |
| **→ ta→en** | Tamil→English translation for the English-only classifier only (interim `Helsinki-NLP/opus-mt-dra-en`; IndicTrans2 is gated behind an HF login); the original text is kept for every other stage | working, interim model | `src/fir/translate/` |
| **C  Statute-ID** | TF-IDF + one-vs-rest logistic on all 66k ILSI documents (the classifier of record); InLegalBERT head trained and measured (loses); IPC→BNS map with the high-confidence-only rule; BNSS First Schedule parsed from the gazette (438 keys); element-wise justification for 29 sections with English + Tamil cues; the same cues run on the original text as a recall safety net ("also consider"); BNS section text from the gazette shown beside every suggestion | working | `src/fir/statute/` |
| **D  IF-1 record** | Pydantic record in parity with the canonical JSON Schema; decision → record bridge; grounded narrative composer (Tamil + English) with a faithfulness gate; renderer for the fixed 15-item form with machine-filled slots annotated | working (narrative is deterministic v0, not an LLM) | `src/fir/schema/`, `src/fir/instantiate/` |
| **E  Verification** | FastAPI (text, audio, statute lookup, health); officer page: microphone recording or audio upload, typed text, demo-mode URLs, route + sections + BNS text + element checks + "also consider" + flags + annotated form, statute lookup box, "Mark verified" disabled by design; append-only audit log of every draft (hashes, never text) | v0 (React workspace planned) | `src/fir/serving/` |
| Eval harness | statute baselines, ensemble, truncation ablation, ASR baseline/profile/sweeps, translation eval, extraction eval, `summarize` → RESULTS.md | working | `harness/` |

Milestone status against PLAN.md (27 milestones): **10 working, 11 partial/v0, 6 not started**
(figure `assets/r3_milestones.png`). Not started: guided-interview TTS, LLM structured extraction,
print-faithful PDF, audit *versioning*, fine-tuning pass, human gold set.

Size: ~11,000 lines of Python across `src/`, `harness/`, `tests/`; **376 tests** pass in ~25 s without
model weights (`bash scripts/run.sh test -q`); ruff clean.

---

## 4. Data and models actually used

| Asset | Role | Size / notes |
|---|---|---|
| ILSI (Zenodo) | statute-ID train/dev/test: 42,835 / 10,200 / 13,039 English court fact-statements, 100 IPC sections | the only labelled statute data; IPC-era; domain shift to spoken complaints is the central risk |
| FLEURS `ta_in` test | ASR yardstick: 60 clips, 742.7 s (read Wikipedia sentences, not complaints) | also joined with `en_us` for the translation eval (336 sentence pairs) |
| `data/mapping/ipc_bns_map.csv` | IPC→BNS conversion, 100 rows: 69 auto-applicable, 29 flagged for review, 7 map outside the BNS | built from a third-party pocket directory, **not** the MHA gazette — hence the strict rule |
| `data/statutes/bnss_schedule1.csv` | cognizability: 438 BNS section keys from 461 printed rows of the BNSS First Schedule Part I (gazette pp. 158–188): 277 cognizable, 135 non-cognizable, 26 conditional | parsed from the official gazette PDF with page citations; **awaits a legal read-through** |
| `data/statutes/bns_2023.jsonl` | all 358 BNS sections: heading, chapter, text, illustration-free operative text | parsed from the BNS gazette PDF |
| `tests/fixtures/extraction_gold.jsonl` | 28 hand-authored adversarial complaints for the extractor | smoke set, not a benchmark |
| Models | `Systran/faster-whisper-large-v3` (3.1 GB, int8 → 1.6 GB VRAM); `law-ai/InLegalBERT` (534 MB); `Helsinki-NLP/opus-mt-dra-en` (621 MB) | all cached locally; nothing downloaded at run time |

Not used yet (planned): Qwen2.5-7B 4-bit as the resident LLM (extraction B, element verifier,
narrative), Qwen2.5-14B for synthetic data (does not fit 8 GB), MuRIL NER, BGE-m3 retrieval,
IndicWhisper / IndicConformer, Kathbath / Common Voice for fine-tuning.

---

## 5. Results — the only numbers that exist (from `artifacts/reports/RESULTS.md`)

### 5.1 Statute identification (ILSI test, thresholds tuned on dev)

| training set | model | micro-F1 | macro-F1 | labels used /100 |
|---|---|---:|---:|---:|
| 8k subsample | TF-IDF + OvR logistic | 0.5988 | 0.4278 | 99 |
| 8k subsample | InLegalBERT (2 ep) | 0.2846 | 0.1811 | 79 |
| 8k subsample | InLegalBERT, head LR ×10 | 0.3530 | 0.2398 | 89 |
| **full 42,835** | **TF-IDF + OvR logistic** | **0.7000** | **0.5805** | 100 |
| full | InLegalBERT (2 ep) | 0.4636 | 0.3984 | 97 |
| full | InLegalBERT (4 ep, head LR ×10, head+tail truncation) | 0.5256 | 0.4594 | 100 |
| full | Ensemble α=0.8·TF-IDF + 0.2·BERT | 0.7006 | 0.5839 | — (gain +0.0006) |

Truncation ablation: cutting inputs to 512 wordpieces keeps 24.1% of the text and costs TF-IDF
0.0516 micro-F1 (8.6% relative) — far less than the encoder gap, so truncation does not explain it.
**Conclusion: TF-IDF is the classifier; the encoder's future role is element verification.**
Figure: `assets/r3_statute_f1.png`.

### 5.2 ASR (faster-whisper large-v3 int8, CUDA, FLEURS ta_in test, 60 clips, 742.7 s)

| decoding | WER | CER | median clip WER | RTF | guard-flagged clips |
|---|---:|---:|---:|---:|---:|
| library default: temperature fallback ladder [0…1.0] (before) | 57.25% | 22.37% | 50.0% | 1.81 | 2 |
| one pass at T=0, sequential 30 s window | 53.49% | 15.82% | 48.1% | 0.51 | 1 |
| **one pass at T=0, batched VAD chunks (bs=8) — pipeline default** | **51.41%** | **15.05%** | 45.5% | **0.46** | 2 |

Profile on 8 clips (finding 6f): the fallback ladder re-decodes low-confidence windows up to six
times — RTF 3.06 vs 0.57 at beam 1 (3.20 vs 0.65 at beam 5) with equal or better WER. On a 148 s
complaint-length recording, batched VAD-chunk decoding beat the sliding window on both axes: RTF 0.31
→ 0.10, WER 81% → 49% (beam 5). Figures: `assets/r3_asr.png`, `assets/r3_asr_profile.png`.

Decoder brakes on the replay loop (finding 6g, negative result): `repetition_penalty 1.1` → WER 53.81%,
1 flagged; `1.2` → WER 57.66%, 0 flagged but median clip WER 45.5% → 57.9%; `no_repeat_ngram_size 5`
made the other clips worse and did not clear clip 1916. **Defaults kept; the transcript guard is the
defence.**

Guard facts: clip 1916 (45.9 s) loops (WER 178%, 41% of trigrams duplicated at the calibration
run); clip 1721 loops; foreign-script drift (Cyrillic) seen once. Tamil is agglutinative, so per-clip
WER above 100% is not by itself fabrication — **always show CER beside WER.**

### 5.3 Translation ta→en (FLoRes/FLEURS 336 sentence pairs, Wikipedia register)

`Helsinki-NLP/opus-mt-dra-en`: **chrF 35.2, BLEU 10.5**, 0.30 s/sentence on CUDA. For scale, published
FLoRes ta→en is ~50 chrF (NLLB-600M) and 55–60 (IndicTrans2). Translation quality bounds statute-ID
quality for Tamil input; the cue scan on the original text is the safety net (finding 5c).

### 5.4 Extraction (rule-based floor, 28 gold documents, 63 gold values)

Per slot — date 18/18, time 16/16, amount 17/17, phone 7/7, vehicle 5/5: **P = R = F1 = 1.000, zero
false positives.** A smoke set, not a benchmark; the property that matters is zero false positives (a
wrong value on the form invites trust; a blank prompts a question).

### 5.5 Legal knowledge base

- IPC→BNS map: 100 rows; 69 auto-applicable; 29 held for review; 7 non-BNS targets.
- BNSS First Schedule: 438 keys (277 cognizable, 135 non-cognizable, 26 conditional). The hand-coded
  stub used until 13 Sep was wrong on **4 of 48** entries — BNS 126, 223, 296, 329 are cognizable in
  the gazette, non-cognizable in the stub (all in the "would have refused an FIR" direction) — and hid
  two conditions: s.85 cruelty is cognizable only when reported by the aggrieved woman or a relative;
  s.303(2) theft is non-cognizable below ₹5,000. Figure: `assets/r3_schedule.png`.
- BNS text: 358/358 sections parsed with headings and chapters.

### 5.6 Cue scan (element-wise justification as a recall net)

29 sections covered with English + Tamil cues. A 22-complaint battery (`tests/test_cue_battery.py`)
pins which sections must fire and which tempting wrong ones must not (a threat to kill is 506, not
307; a road death is 304A, not 302). No precision/recall number is reported for it — it is a test
battery, not a benchmark; say "22 pinned cases", not a percentage.

---

## 6. Caveats you must carry onto any slide that shows a number

1. ILSI is English court text; complaints are short spoken Tamil. The TF-IDF classifier **abstains on
   short complaints** (its top-5 for our own two-sentence theft sample does not contain theft), so a
   lower threshold would apply *wrong* sections. The cue scan carries short inputs; in-domain synthetic
   data is the planned fix. Never claim the classifier "works on complaints".
2. WER 51% is zero-shot on read Wikipedia sentences, not police speech. It is honest and it is high;
   the answer is provenance + guards now, IndicWhisper / a code-switch fine-tune next.
3. The translator is an interim model (chrF 35); IndicTrans2 needs a gated download. One real
   failure — "வரதட்சணை" (dowry) → "relief" — is why the cue scan exists (finding 5c).
4. The legal tables are parsed from the official gazette with page citations but have **no legal
   sign-off** yet; the form says so on its Basis line.
5. The narrative composer and element checks are rule-based v0; the LLM versions are not built.
6. Extraction 63/63 is on our own 28 documents. Do not present it as a benchmark result.
7. The GPU is 8 GB, not the 16 GB in the locked decisions — a recorded finding, not a secret.

---

## 7. The findings — the story the slides should tell (numbers match `PROGRESS.md`)

| # | Finding | Evidence | What we did |
|---|---|---|---|
| 1 | GPU is 8 GB, not 16 | torch reports 8.5 GB; Qwen-14B-Q4 ≈ 9 GB | `configs/models.yaml` carries a `fits` flag per model; 7B vs 14B decision deferred to a mains-power benchmark |
| 2/3 | Blackwell needs CUDA 12.8; CTranslate2 aborts at C level without cuDNN | first runs crashed silently | torch cu128; libraries preloaded before import |
| 5 | Domain shift is measured | classifier abstains on short complaints | cue scan as recall net; synthetic data planned |
| 5b | The classifier is English-only | Tamil input → no predictions | translation node (interim model) feeding the classifier only |
| 5c | A translation error turned a dowry-cruelty complaint into a CSR | "dowry" → "relief" on the real server | bilingual cue scan on the original text forces officer review; test reproduces the failure |
| 6 | Tamil ASR is genuinely hard for Whisper | WER 55–57% zero-shot | keep int8; report CER; plan model-side upgrades |
| 6b | Whisper fabricated text on 2/60 clips | replay loop 226% WER; Cyrillic drift | transcript guards calibrated on the baseline; flagged → not auto-resolvable |
| 6c | For Tamil, WER > 100% ≠ fabrication | clip 1731: WER 150%, CER 13% | CER always shown beside WER |
| 6e | The WSL VM rebooted under full-corpus jobs | 6 deaths; Python heap > ~2 GB | chunked builds; detached launches; logs on the Windows file system |
| 6f | ASR was 3–4× slower than real time because of the temperature-fallback ladder | RTF 3.06 → 0.57 with one pass at T=0; batched VAD chunks RTF 0.31 → 0.10 on a long recording | both are the pipeline defaults; 60-clip WER 57→51%, CER 22→15%, RTF 1.81→0.46 |
| 6g | Decoder brakes clear the loop only by hurting every other clip | repetition_penalty 1.2: 0 flagged but median WER 45→58% | rejected; guard stays; fix is model-side |
| 7 | InLegalBERT head collapsed to 6/100 labels | loss flat at 0.142, macro-F1 0.018 | positive-class weighting, higher head LR — then it still loses to TF-IDF |
| 8 | Truncation is not why the encoder loses | ablation: 8.6% relative | ruled out; conclusion stands |
| 9 | The hand-coded cognizability stub was wrong on 4/48 | gazette parse vs stub | Schedule parsed from the gazette; `conditional` class; theft split resolved by the extracted value |

The one-sentence arc: *a cheap sparse baseline beat the domain encoder, the legal tables had to come from
the gazette and not from memory, and every safety guard in the system exists because a real failure
was reproduced first.*

---

## 8. Demo (what a panel sees) — full script in `docs/REVIEW3_DEMO.md`

Start: `wsl -d Ubuntu -e bash -lc "cd /mnt/c/Users/madha/Downloads/fir-tn-capstone && FIR_PREWARM=1 bash scripts/run.sh serve"`
then `http://127.0.0.1:8000/`. Demo-mode URLs: `/?sample=theft&run=1`, `/?sample=dowry death&run=1`,
`/?sample=spoken Tamil&run=1`, `/?text=<anything>&run=1`.

1. Typed theft complaint with a value → officer review, BNS 303(2) cognizable (₹15,000 > ₹5,000) in "also consider".
2. Remove the value → "cognizability depends on property value"; nothing defaults.
3. Dowry-cruelty complaint → FIR via BNS 80; BNS 85 conditional (who reports); element checks with quoted words; BNS gazette text above the IPC lineage text.
4. Spoken Tamil sample → ITN turns "ஐம்பதாயிரம் ரூபாய்" into ₹50,000.
5. **● Record** a Tamil complaint into the microphone → transcript → same decision, all local.
6. Upload FLEURS clip 1916 → the transcript guard fires; the draft is marked not auto-resolvable.
7. Look up `85` → gazette text + the Schedule's condition, bailable, court.

CLI equivalent: `python -m fir draft --json "..."` (same graph). API docs at `/docs`.

---

## 9. Figures and files for slides

| File | Shows |
|---|---|
| `assets/r3_pipeline_status.png` | the six stages as built, green = working, amber = v0 |
| `assets/r3_milestones.png` | 27 PLAN.md milestones: working / partial / not started |
| `assets/r3_statute_f1.png` | micro/macro-F1: TF-IDF vs InLegalBERT vs ensemble |
| `assets/r3_asr.png` | WER/CER/RTF before and after the decoding fix (60 clips) |
| `assets/r3_asr_profile.png` | the fallback-ladder cost and VAD-chunk batching (8 clips / 148 s recording) |
| `assets/r3_schedule.png` | First Schedule classification pie + the stub's errors |
| `assets/fig_architecture.png`, `fig_pipeline.png`, `fig_dataflow.png`, `fig_mapping.png`, `fig_timeline.png` | Review 2 design figures (still valid as the design) |
| `Review3_Presentation.pptx` / `.pdf` | the 18-slide deck built by `scripts/review3/build_deck.py` |
| `Review2_Presentation.pptx`, `Review2_Report.docx`, `Literature_Review.docx` | Review 2 deliverables (literature review cites 16 recent papers) |
| `artifacts/reports/RESULTS.md` | every table; regenerate with `bash scripts/run.sh results` |
| `PLAN.md`, `ARCHITECTURE.md`, `IMPLEMENTATION.md`, `DATASETS.md`, `OPEN-QUESTIONS.md` | the planning documents (design, schema, risks, data verification) |

Regenerate figures/deck on the Windows side: `python scripts/review3/make_figures.py` then
`python scripts/review3/build_deck.py`; export PDF from PowerPoint.

---

## 10. Repository map

```
src/fir/            the package (asr, extract, translate, statute, schema, instantiate, orchestrator, serving, cli)
harness/            experiments and evaluation runners; summarize.py -> RESULTS.md
tests/              376 pytest tests + fixtures (golden_slice.json, extraction_gold.jsonl, asr_quality_cases.json)
configs/            pipeline.yaml (every knob), models.yaml (fits flags), datasets.yaml
data/mapping/       ipc_bns_map.csv (+ the pocket-directory PDF it came from)
data/statutes/      gazette PDFs, bnss_schedule1{,_rows}.csv, bns_2023.jsonl, README.md (provenance)
scripts/            downloads, gazette parsers, GPU queues, run.sh (make-less target runner), review3/ builders
docs/REVIEW3_DEMO.md  demo script, pre-flight, fallbacks, likely panel questions
artifacts/reports/  RESULTS.md (committed); JSON reports and logs are local-only (gitignored)
assets/             figures
```

Run anything: WSL2 Ubuntu, venv `~/.venvs/fir-tn` (Python 3.11, torch cu128). `bash scripts/run.sh <target>`
mirrors the Makefile (`test`, `lint`, `serve`, `results`, `schedule`, `bns-text`, `slice`…). Windows-side
Python only runs the deck/figure builders (Smart App Control blocks `pyarrow`, so no HF `datasets`).

---

## 11. Rubric mapping and what the team must still fill in

| Rubric parameter | Evidence in this repo | Still needed from the team |
|---|---|---|
| 1 Follow-up on Review 2 + progress vs plan | `assets/r3_milestones.png`; PROGRESS.md dated log | **the panel's Review 2 comments and the action taken on each** (slide 3 is placeholders) |
| 2 Implementation ≈50% | §3; live demo; 376 tests | — |
| 3 Technical accuracy / best practices | invariants as tests; gazette-cited tables; sparse baseline in every comparison; negative results recorded | — |
| 4 Interim results | §5 tables + figures | — |
| 5 Problem-solving | §7 findings | — |
| 6 Report progress | `Review3_Report.docx` (Ch 1–3 + §3.8 revisions + Ch 4 draft) exists though not required for this review | — |
| 7 Presentation / individual responsibility | deck; demo | **the contribution table** — see below; **AI-tool acknowledgement** wording; a publication venue for Review 4 (2 marks only against proof of submission) |

**Individual contribution — read before writing that slide.** The Review 2 plan split the work by
stage (Madhav: statute-ID and instantiation; Nischay: ASR; Rohit: extraction and synthetic data).
The implementation in this repository between 20 Aug and 15 Sep was carried out by Madhav using an AI
coding assistant, as the commit history and `PROGRESS.md` show. The rubric grades *demonstrated*
contribution per member and the guidelines require AI use to be acknowledged — present the split as it
actually happened and agree the wording with the guide. Do not have a slide claim work that the log
does not show.

Suggested acknowledgement line (confirm with the guide): *"Implementation was carried out with the
assistance of an AI coding assistant (Claude Code); all design decisions, data choices, experiments and
their interpretation were reviewed and are owned by the team."*

---

## 12. Timeline (absolute dates)

- 17 Aug 2026 — datasets verified live (FLEURS, Kathbath, ILSI, InLegalNER, BNS/BNSS text); DATASETS.md, IMPLEMENTATION.md.
- 18–19 Aug — Review 2 deliverables (deck, report, literature review); Review 2 panel 19 Aug.
- 20 Aug — repo skeleton, WSL env, first vertical slice (TF-IDF + InLegalBERT baselines, Whisper baseline, LangGraph graph, golden tests); 8 GB GPU finding.
- 10–13 Sep — full-corpus statute runs on the GPU queue (encoder settled); ITN; hallucination guards; Stage D record/narrative/form; FastAPI + officer page; translation node; cue scan (finding 5c); gazette parsing of the First Schedule and BNS text; element cues to 29 sections; Review 3 deck.
- 15 Sep — ASR decoding fix (finding 6f) re-baselined; decoder-brake sweep (6g, negative); microphone / upload on the officer page; audit log; demo script; this file.
- 16 Sep — Review 3 panel. Next: Review 4 (12–16 Oct), final panel 21 Oct.

## 13. Roadmap to Review 4/5 (what to promise, carefully)

1. In-domain data: synthetic Tamil complaint→IF-1 pairs with a local LLM (the 7B/14B decision is open because of 8 GB); a self-recorded human gold set.
2. Model-side ASR: IndicWhisper / a Tamil-fine-tuned Whisper benchmark; then a code-switch LoRA.
3. Translator upgrade: IndicTrans2 (gated login) or NLLB-600M.
4. LLM element verifier and narrative (Qwen2.5-7B 4-bit) against the same schema; RAG over the parsed BNS text.
5. React verification workspace with accept/override per section and audit versioning; print-faithful IF-1 PDF.
6. Legal read-through of the Schedule CSV and the 29 flagged mapping rows.
7. Publication: a venue to be agreed with the guide (Review 4 awards marks only against proof of submission).

## 14. Glossary

FIR — First Information Report · IF-1 — CCTNS Integrated Investigation Form 1 (the fixed TN FIR form) ·
BNS / BNSS — Bharatiya Nyaya Sanhita / Bharatiya Nagarik Suraksha Sanhita 2023 (replaced IPC / CrPC) ·
IPC — Indian Penal Code 1860 (the labels in ILSI) · CCTNS — Crime & Criminal Tracking Network & Systems ·
CSR — Community Service Register (non-cognizable matters) · cognizable — police may register and
investigate without a magistrate's order · ILSI — Indian Legal Statute Identification corpus ·
ITN — inverse text normalisation (spoken numbers → digits) · WER / CER — word / character error rate ·
RTF — real-time factor (processing time ÷ audio duration; < 1 is faster than real time) ·
TF-IDF — sparse lexical features · InLegalBERT — a BERT pre-trained on Indian legal text ·
micro / macro-F1 — F1 pooled over all predictions / averaged over the 100 sections ·
LangGraph — the graph runtime the stages run in · cue scan — bilingual keyword ingredients per section,
used as a recall net · "conditional" — a Schedule entry whose cognizability depends on a fact.
