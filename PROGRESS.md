# PROGRESS — session handoff

> Update this at the END of each working session so any NEW session — a fresh Claude Code run, a
> chat, or a Cowork session — can resume instantly without re-explaining anything.
> Fresh-session read order: CONTEXT.md → this file → README.md → IMPLEMENTATION.md.

## Current phase
**IMPLEMENTATION — Stage A/B-floor/C/D backbone complete, end to end.** The graph now runs from a
transcript (or audio) through rule extraction, statute-ID, IPC→BNS, cognizability, IF-1 record
assembly, grounded narrative composition, and deterministic rendering of the fixed IF-1 form.
Learned extraction (Stage B proper), the LLM narrative, and the officer UI (Stage E) are not built.
**Madhav is building solo (decided 2026-09-13).** Planning + Review-2 deliverables remain complete.

## Last updated
2026-09-13

---

## Done this session (2026-09-13) — solo build, ASR fix + Stage D backbone

**368 tests pass** (`make test`, ~20s, no weights), **ruff clean** (`make lint`; E/F/W/I, E501 ignored). ~7,700 lines added across 28 new modules. Everything
below is wired into the LangGraph graph and exercised by `make report`; the verification page was
browser-checked against the real server at the end of the session (elements, statute text, cue
hits, translation row all rendering).

### ASR: hallucination guard + decoder fix (finding 6b, now 6b/6c)
- Re-examined the 3 "hallucinated" clips: they were **three different things** — a decoder replay loop
  (1916), a language drift into Cyrillic/Greek (1996), and a **metric artefact** (1731: WER 150% but
  CER 13%, just an agglutination boundary split — not a fault at all). See finding 6c.
- `src/fir/asr/quality.py`: transcript guards for repetition (duplicated-trigram fraction) and foreign
  script (absolute count of non-Tamil/Latin letters). **Thresholds calibrated on the real 60-clip
  baseline, not guessed**: every clean clip scores exactly 0% duplication; the replay scores 41%;
  Cyrillic count 8 vs 0 elsewhere. Latin is allowed — real complaints are code-switched.
- `condition_on_previous_text=False` in `pipeline.yaml` — **predicted** to stop the 30s-window replay;
  a controlled rerun showed it does not (finding 6b). Kept as standard practice; the guard is the defence.
- Flags propagate: `Transcript.flags` → graph `asr_flags` → `StatuteDecision.asr_flags` →
  **`is_auto_resolvable` is False for any flagged transcript**, whatever the statute result says.
- Fixture strings are the *actual* Whisper output for those clips (`tests/fixtures/asr_quality_cases.json`).
- Rerun of the 60 clips with the guard live: 2 of 60 flagged, both correctly (finding 6b).

### Stage D: the IF-1 record, bridge, renderer, and composer
- `src/fir/schema/if1.py` — Pydantic form of ARCHITECTURE.md §5, `extra="forbid"` everywhere.
  `tests/test_if1_schema.py` walks the canonical JSON Schema (`if1_v1.schema.json`, extracted from
  ARCHITECTURE.md) and fails if any property exists in one but not the other — 23 top-level, 7 `$defs`,
  5 inline blocks, all in parity. Invariants: `narrative_is_grounded` (the faithfulness gate),
  `is_fileable` (True only when the officer sets `verified`; nothing in `src/` does).
- `src/fir/instantiate/from_decision.py` — `StatuteDecision` → `acts_sections` + `cognizability`.
  Held-back sections become `mapping_ambiguous=True` candidates with a capped score, **recorded not
  dropped**. CSR route sets `status=csr_advisory`. Never sets `verified`.
- `src/fir/instantiate/render.py` — the fixed 15-item CCTNS IF-1, deterministic. Items 13–15
  (action taken / signature / dispatch) are **always blank**: they are the officer's. Empty slots render
  as visible `________` because on a fixed form a blank is information. `annotate=True` tags every
  machine-filled slot with status/confidence/provenance count for the verification screen.
- `src/fir/instantiate/narrative.py` — item 12 composed from templates, **one sentence per filled
  slot, each carrying that slot's provenance**. The composer is structurally incapable of writing an
  ungrounded factual sentence, so the faithfulness gate passes by construction. English is
  authoritative; **Tamil templates need native-speaker review before real use.**

### Stage A: Tamil inverse text normalisation (ITN)
- `src/fir/asr/itn.py` — spoken numbers → integers, Tamil + English, with offsets. Tamil numerals
  compound with **sandhi** (ஐம்பது + ஆயிரம் → ஐம்பதாயிரம், இருபது + ஐந்து → இருபத்தைந்து), so it is a
  small grammar: a greedy lexer over a 206-form stem table (standalone, joining-with-pulli,
  joining-without-pulli, and *derived* vowel-sign forms — ஐந்து → ைந்து — generated rather than
  hand-listed) plus an accumulator with Indian grouping (crore > lakh > thousand, strictly
  decreasing or it is two numbers). Colloquial forms (அஞ்சு, நாலு, ரெண்டு), ordinals (-ஆம்/-ஆவது),
  English ordinals derived (`fifteenth`, `twentieth`).
- **68 tests, including 9 false-positive cases** — the stems are short (ஐ, மு, நா) so whole-word
  matching is what keeps ஒருவர், நாள், முதல், இருந்து, ஐயா, ஆறுதல், எண்ணம் from becoming numbers.
  Known ambiguity pinned in a test: ஆறு is both "six" and "river"; the amount extractor therefore
  requires an adjacent currency word before it accepts a spoken number.
- Wired into **all three** numeric extractors via `NormalisedText` — one digit-normalised view of
  the transcript with an offset map back, so `பத்து மணி`, `மே எட்டாம் தேதி` and `ஐம்பதாயிரம் ரூபாய்`
  hit the same regexes as their digit forms while provenance still points at the spoken words.
  Spoken values carry confidence 0.85 vs 0.90 for typed digits. One interaction bug caught and
  fixed: a bare multiplier after a digit ("5 lakh") is a scale suffix, not a number — ITN must
  leave it for the amount regex or "5 lakh" becomes "5 100000".
- **Year-less spoken dates** ("மே எட்டாம் தேதி", "the fifteenth of June") now extract as ISO
  `--MM-DD` at confidence 0.6 with an officer note — spoken complaints routinely omit the year and a
  partial the officer can complete beats a blank. Never combined into an interval with a full date.
- Gold smoke set extended to 20 docs with spoken-register cases and a trap ("ஆறு பேர்", "ஒருவர்",
  "நாள்" — three number stems, zero numbers): **45/45, zero false positives.** Extended again to 28 docs
  with the cue-battery complaints (slash dates, "3.15 pm", "two lakh rupees", "ஒரு லட்சம் ரூபாய்", a
  single-letter plate series "TN 45 Z 9090", and "9-year-old" as a number that is *not* a slot):
  **63/63, zero false positives.**

### Stage C: statute text next to the number, and a corroboration signal
- `src/fir/statute/statute_text.py` — lookup + TF-IDF cosine retrieval over ILSI's 100 IPC section
  texts (`secs.jsonl`, median 300 chars). First step toward the element verifier (PLAN P3): the
  retriever that will fetch statute text for an LLM to check elements against.
- Every `SectionSuggestion` now carries `statute_text` (the words of the law, not just a number) and
  `retrieval_rank` — where the section lands when the statute texts are retrieved against the
  narrative. **This is a disagreement detector**: on "harassed for dowry… found dead", 304B ranks #1
  and 498A #2 by the statute's own wording, while 302 (murder) and 34 (common intention) have *no*
  lexical support — shown on the UI as "no textual support", not hidden. Held-back review flags
  carry the text too.
- **Two retrieval defects found and fixed while wiring translation.** (1) A complaint mentioning
  "fifty thousand rupees" retrieved sections 448/342/323 — by the size of their *fine clause*
  ("…fine which may extend to one thousand rupees"). Punishment boilerplate is shared by nearly every
  section and is pure noise for fact→offence matching; it is now a stop list (35 terms; **not**
  "death", which is content in 302/304B). (2) "cheated" never matched "cheats" — a light suffix
  stemmer now unifies inflections, and the stop list is stemmed too or the stems slip past it.
  After both: "committed theft" → 379 at 1.0, "cheated… false promise" → 417/419, and "stole my
  phone worth fifty thousand rupees" → **nothing**, which is honest: lexically there is no overlap
  ("stole" ≠ "theft"). That vocabulary gap is the argument for BGE-m3 dense retrieval (PLAN P3).
  `retrieval_rank` is computed on the English view when a translation exists.
- BNS text will replace the IPC text once `data/statutes/` is populated; the interface is unchanged.

### Stage C: element-wise justification, v0 (PLAN P3 — the research core's distinctive output)
- `src/fir/statute/elements.py` — hand-authored element lists for the 9 common FIR sections
  (379/380 theft, 302 murder, 304B dowry death, 498A cruelty, 323 hurt, 506 intimidation, 420
  cheating, 354 outraging modesty), each ingredient with English + Tamil lexical cues. Fills
  `SectionCandidate.elements` in the IF-1 record; renders on item 2 of the form and in the UI.
- **Two honesty rules**: it never says *no* (a missing cue is absence of evidence, not evidence of
  absence — it says *unclear* and names what the officer must establish), and every *yes* quotes the
  matched words with offsets. Uncovered sections say "elements not analysed", not nothing.
- Already informative on the demo: for "married three years ago… harassed for dowry… found dead by
  hanging", **304B shows 4/4 ingredients evidenced while 302 shows death but intent *unclear***
  ("establish: intention or knowledge: weapon, manner, prior threats?"). That is the shape of the
  final output — the LLM verifier replaces the cue lists, against the same schema.

### Translation node (finding 5b): Tamil → English for the classifier only
- `src/fir/translate/indictrans.py` + graph node `translate` between `extract` and `statute_id`.
  Produces `narrative_en`, which **only the classifier reads**; extraction, elements and the narrative
  composer keep the original transcript so provenance offsets stay valid and Tamil cues stay
  first-class. Runs only when Tamil-dominant (≥30% of alphabetic chars). No model on disk → no-op
  with an explicit error, graph still completes.
- Three backends, inferred from the model id: **IndicTrans2** (best; 800 MB; MIT; **gated** — needs
  *your* `huggingface-cli login` + terms acceptance, which I cannot do), **NLLB-200-distilled-600M**
  (ungated; 2.5 GB; CC-BY-NC — above my download-ask threshold, so not fetched), and
  **opus-mt-dra-en** (ungated; 621 MB; Apache-2.0; weakest) — **the interim default, downloaded**, so the
  node is provable end to end. Swapping is a one-line config change (`translate.model`).
- Wired into serving (`/v1/health` reports the translator; `DraftResponse.narrative_en`), the UI
  ("Classifier read" row), and the CLI. 6 hermetic tests with a stub translator.
- **Measured**: chrF 35.2 / BLEU 10.5 on 336 FLEURS/FLoRes pairs (`harness/run_translation_eval.py`,
  `make results` renders it) — ~20 chrF behind IndicTrans2. It turned "dowry" into "relief" on the
  first real complaint, which produced finding 5c and the cue-scan safety net.

### Stage B floor: rule extraction with provenance
- `src/fir/extract/rules.py` — dates (numeric d/m/y with the m/d reading offered as an alternative,
  word dates in English + Tamil month names), times (am/pm, o'clock, மணி, காலை/மாலை/இரவு), INR amounts
  (Indian digit grouping, lakh/crore, ₹/Rs/ரூபாய்), Indian mobiles, vehicle plates. Every span has
  character offsets → `Provenance`. Spelled-out Tamil numerals with sandhi are **deliberately not
  parsed** — that is ITN, owned by the ASR stage.
- `src/fir/extract/fill.py` — fills a slot **only when exactly one reading exists**; otherwise leaves
  it empty with the candidates in `officer_note`. Two phone numbers → ask, don't guess.
- 47 tests across 20 surface forms.

### Cognizability: the real BNSS First Schedule replaces the stub (finding 9)
- `scripts/build_bnss_schedule.py` parses Part I of the First Schedule ("Offences under the BNS",
  gazette pp. 158-188) out of the official BNSS gazette PDF (MHA copy, `data/statutes/`) into
  `bnss_schedule1_rows.csv` (461 verbatim rows with page numbers, for the legal reviewer) and
  `bnss_schedule1.csv` (438 section keys: 277 cognizable, 135 non-cognizable, 26 conditional, 0 parse
  gaps). Coordinate parser: the table has no ruling lines and the two classification columns drift
  by up to 35 pt between pages, so they are located per row from the words that can only begin a
  classification cell. `make schedule` rebuilds and prints the disagreements with the old stub.
- `cognizability.py`: new `Cognizability.CONDITIONAL` for entries the Schedule makes depend on a fact
  ("According as offence abetted is cognizable...", s.85 "Cognizable if information ... is given by
  the person aggrieved", s.303(2) theft non-cognizable under Rs 5,000). Routes to the officer like
  `unknown` but the rationale quotes the Schedule's words. A machine-readable `condition` column
  resolves the one split that is on an extracted fact: `property_value_inr<5000=non_cognizable;
  else=cognizable` — the graph passes the single unambiguous extracted amount as context, so a
  theft *with* a stated value is FIR or CSR and a theft *without* one asks the officer.
  Composite mapping targets (`324(4),(5)`) resolve only if the parts agree; base numbers the
  Schedule does not print (`103`, `80`) are derived only where every sub-clause agrees.
- Cue-scan safety net now raises the route for any hit that is not settled non-cognizable
  (conditional/unknown included) — otherwise s.85 turning conditional would have re-opened the
  finding-5c hole. `tests/test_bnss_schedule.py` (41 tests) pins file integrity, 25 classifications
  a lawyer can check against the printed Schedule, and the loader logic. Browser-verified.

### Statute text: the officer now reads the BNS words, not ILSI's IPC text
- `scripts/build_bns_text.py` parses all **358 BNS sections** out of the BNS gazette PDF (102 pp.):
  marginal heading, chapter, full text, and `operative_text` with the worked illustrations stripped
  (an illustration block runs from "Illustrations." to the next sub-section / Explanation / Exception /
  proviso). Numbering must come out as 1..358 with no gaps or the build fails. Marginal citations of
  other Acts ("45 of 1860.") and the e-gazette signature block are filtered. `data/statutes/bns_2023.jsonl`.
- `src/fir/statute/bns_text.py` — read side: `text_for("BNS 303(2)")` returns the "(2) ..." paragraph
  (so the theft suggestion shows the punishment clause with its Rs 5,000 proviso, not the definition);
  composite targets get each clause; a clause the gazette does not number falls back to the section.
- `StatuteDecision`: `bns_heading` / `bns_text` on suggestions and review flags; the IPC text stays as
  `statute_text` and the UI labels it lineage. `fir check` reports the corpus. 20 tests. Browser-verified
  on the dowry sample (BNS 80 / 85 wording rendered above the IPC 304B / 498A text).
- This is also the corpus the element verifier (PLAN P3) reads against — the retrieval in
  `statute_text.py` still indexes ILSI's IPC texts because that is what the classifier's labels are;
  moving retrieval to BNS text is a one-line swap once the IPC→BNS map is complete.

### Element cues: coverage 9 → 29 sections
- `elements.py` second batch: 392 robbery, 384 extortion, 406 breach of trust, 411 receiving stolen
  property, 341/342 restraint/confinement, 363/366 kidnapping, 324/325 hurt by weapon / grievous, 307
  attempt to murder, 306 abetment of suicide, 279/304A rash driving / death by negligence, 427 mischief,
  447/448 trespass / house-trespass, 509 insulting modesty, 294 obscene acts, 419 personation (incl.
  OTP/bank-caller fraud, also added to 420). Every element has EN + TA cues and an officer note.
- Two precision fixes found by a 22-complaint battery (`tests/test_cue_battery.py`): "threatened to kill
  me" must not evidence 307 (same lesson as 302 — a threat is 506), and Tamil "போனை" (inflected
  "phone") did not match the "போன்" cue — match stems, not citation forms, for Tamil nouns.
- The battery pins both directions: sections that *must* fire on each complaint and the tempting wrong
  ones that must not (road death → 304A not 302; stalking → 509 not 354/376).

### Graph
`(audio→asr) → extract → statute_id → ipc_bns → cognizability → instantiate → END`. New state:
`extracted_spans`, `fir_record`, `if1_text`, `fill_report`, `narrative_grounded`, `asr_flags`.
`make report` now prints a rendered IF-1 from the real TF-IDF model.

### Serving + Stage E v0 + CLI
- `src/fir/serving/app.py` — FastAPI, localhost only. `POST /v1/complaint/{text,audio}` returns
  decision + record + rendered form + flags. `GET /v1/health` answers without models; a box without the
  TF-IDF artifact gets a 503 with the reason, not a 500. **No endpoint can set `verified`.**
- `src/fir/serving/static/index.html` — the officer verification screen, one static page driving the
  API: route badge, BNS sections with IPC lineage, held-back mappings, filled/ambiguous slots, ASR
  flags, faithfulness gate, annotated form with machine-filled slots highlighted, and a **"Mark
  verified" button that is disabled by design** with the reason stated. Served at `/`. React later;
  the API contract does not change.
- `GET /v1/statute/{bns_section}` — reference lookup: gazette heading/text of the section or named
  sub-section plus the First Schedule entry (cognizable / conditional with the Schedule's words,
  bailable, court). The page has a "Look up a section" box on it. Read-only; not a decision.
- `python -m fir draft|check|mapping` CLI. `make serve`.

### Eval harness (IMPLEMENTATION.md §6 "in parallel from day 1")
- `harness/run_extraction_eval.py` + `tests/fixtures/extraction_gold.jsonl` — per-slot P/R for any
  extractor exposing `extract_all`. 15 hand-authored adversarial docs (impossible dates, bare numbers,
  account numbers shaped like phones, two-phone cases, Tamil year-first dates). Rules extractor:
  **36/36, zero false positives.** Smoke set, not a benchmark — the real Stage B eval needs synthetic data.
- `harness/summarize.py` → `artifacts/reports/RESULTS.md` (`make results`): every JSON report rendered
  as the tables the Review-3 report needs. Recomputes nothing.
- Encoder options ready for the next experiments, both default-off so the running full-corpus job stays
  a clean data-effect measurement: `head_lr_multiplier` (finding 8 hypothesis #4) and
  `truncation: head_tail` (Sun et al. 2019). The `head_tail` test caught a real `toks[-0:]` slice bug.

---

## Done (2026-08-20) — Task 1 + Task 2

---

## Done this session (2026-08-20) — Task 1 + Task 2

### Environment (Task 1)
- WSL2 Ubuntu 24.04, **Python 3.11.15** via `uv` (no sudo needed — this box has no passwordless sudo,
  so deadsnakes/apt was not an option). Venv at `~/.venvs/fir-tn` on ext4, **not** `/mnt/c` (9p imports
  are ~10x slower). Bootstrap: `bash environment/setup_wsl.sh`.
- **torch 2.11.0+cu128, CUDA available, sm_120 detected.** Pinned to the cu128 index deliberately —
  see the Blackwell note below.
- Full monorepo layout per IMPLEMENTATION.md §2 (~2,700 lines). `configs/datasets.yaml` +
  `models.yaml` + `pipeline.yaml`. `git init` done; **nothing committed yet** (files are staged).
- **Zero pyarrow surface**, even inside WSL: every loader reads plain JSONL/TSV/tar, and the
  InLegalBERT trainer is a hand-written torch loop rather than HF `Trainer`, so HF `datasets` is never
  imported. Keeps the code runnable natively if that is ever needed.

### Data downloaded and verified live
| | |
|---|---|
| ILSI | **42,834 train / 10,199 dev / 13,038 test** (66,071 docs, 100 IPC sections), 512 MB from Zenodo |
| FLEURS ta_in test | **591 clips** + transcripts, 407 MB |
| law-ai/InLegalBERT | 534 MB, cached |
| Systran/faster-whisper-large-v3 | 3.09 GB, cached |

ILSI shape measured: **avg 3.67 labels/doc, avg 7,180 chars/doc** — long court fact-statements, and
both of those numbers turn out to drive findings below: the label sparsity causes finding 7, and the
document length causes finding 5 (domain shift vs short spoken complaints).

### Vertical slice (Task 2) — results
- **Statute-ID F1 table** (ILSI, 8,000 train / 2,000 test, 100 IPC sections, threshold tuned on dev):

  | model | micro-F1 | macro-F1 | micro-P | micro-R | labels used | train time |
  |---|---|---|---|---|---|---|
  | **TF-IDF + OvR logistic** | **0.5988** | **0.4278** | 0.6569 | 0.5502 | 99/100 | 139s (CPU) |
  | InLegalBERT + pos_weight | 0.2846 | 0.1811 | 0.2523 | 0.3264 | 79/100 | 9.7 min (GPU) |
  | InLegalBERT, plain BCE ❌ | 0.2394 | 0.0175 | 0.2133 | 0.2727 | **6**/100 | 11.8 min (GPU) |

  **The cheap sparse baseline beats the domain-pretrained encoder by 2x.** That is not the expected
  result and it is not (only) a tuning failure — see findings 7 and 8. `pos_weight` fixed the
  all-zeros collapse (macro-F1 0.0175 → 0.1811, labels used 6 → 79) but did not close the gap.

- **Full corpus (2026-09-13)** — TF-IDF on all **42,835** train docs, evaluated on all **13,039** test:

  | train docs | micro-F1 | macro-F1 | micro-P | micro-R | fit time |
  |---|---|---|---|---|---|
  | 8,000 | 0.5988 | 0.4278 | 0.6569 | 0.5502 | 139 s |
  | **42,835** | **0.7000** | **0.5805** | 0.7318 | 0.6709 | 1,400 s (n_jobs=2) |

  5.4x data → **+0.10 micro (+17% rel), +0.15 macro (+36% rel)**. The tail gains most, as finding 8
  predicted. Different test sets (2k vs 13k), so the delta is indicative, not exact. Chunked
  `HashingVectorizer` (2^18 features) rather than `TfidfVectorizer` — finding 6e. **This is the
  baseline of record for statute-ID** until the encoder beats it on the same 13,039 docs.

- **Full corpus, InLegalBERT** (same 13,039 test docs, 2 epochs, pos_weight, head truncation, 52 min):

  | model | train docs | micro-F1 | macro-F1 | dev trajectory |
  |---|---|---|---|---|
  | InLegalBERT | 8,000 | 0.2846 | 0.1811 | 0.296 → 0.296 (flat) |
  | **InLegalBERT** | **42,835** | **0.4636** | **0.3984** | 0.433 → **0.472** (still climbing) |
  | TF-IDF | 42,835 | **0.7000** | **0.5805** | — |

  **The encoder gained +0.18 micro / +0.22 macro from 5.4x data — nearly double TF-IDF's gain** — so
  finding 8's "data-starved" hypothesis is confirmed as the dominant factor. Two epochs is also
  clearly short: dev rose 0.04 between epochs 1 and 2 where the 8k run had flatlined. **TF-IDF still
  leads by 0.24 micro on identical test data.** Next levers, in order: more epochs (4–6), then the
  head-LR and head+tail ablations queued behind this run.

- **Ensemble, full corpus** (`harness/ensemble_eval.py`, p = α·TF-IDF + (1−α)·BERT, α and threshold
  chosen on dev, 13,039 test docs): **the encoder adds nothing.** The sweep is monotonic in α; the
  dev-chosen α=0.9 scores 0.6986 vs TF-IDF alone 0.7000 (−0.0014, noise). At α≈0.6 macro-F1 edges to
  0.5840 vs 0.5805 — a faint tail-complementarity, within noise. Reading: the encoder is not merely
  weaker, it is **redundant with the linear model** — it has learned the surface cues TF-IDF already
  captures and nothing orthogonal. That is the signature of undertraining, not of a wrong model class.
  The ensemble is not worth serving until the encoder learns something the n-grams cannot.

- **Ablations on the 8k slice** (same 2,000 test docs, one variable each, 2 epochs, ~11 min apiece):

  | variant | micro-F1 | macro-F1 | Δ micro | labels used |
  |---|---|---|---|---|
  | InLegalBERT baseline (head, shared LR) | 0.2846 | 0.1811 | — | 79 |
  | + `truncation: head_tail` (Sun et al. 2019) | 0.3170 | 0.2049 | +0.032 | 83 |
  | **+ `head_lr_multiplier: 10`** | **0.3530** | **0.2398** | **+0.068 (+24% rel)** | 89 |

  Finding 8's hypothesis #4 — **the randomly-initialised head is starved at a shared 2e-5** — is the
  single biggest lever found. Head+tail helps modestly, consistent with the truncation ablation's
  ~16%-of-gap estimate.

- **Every lever at once — full corpus, 4 epochs, head LR ×10, head+tail** (`InLegalBERT[e4_ht_hl10]`,
  112 min, same 13,039 test docs):

  | epoch | dev micro | dev macro | Δ micro |
  |---|---|---|---|
  | 1 | 0.4404 | 0.3611 | — |
  | 2 | 0.4991 | 0.4364 | +0.059 |
  | 3 | 0.5218 | 0.4602 | +0.023 |
  | 4 | **0.5341** | **0.4659** | +0.012 |

  **Test: micro-F1 0.5256, macro-F1 0.4594** — +0.062 / +0.061 over the 2-epoch default config, and
  converging (gains halve each epoch; 8 epochs would land ~0.55). **TF-IDF: 0.7000 / 0.5805.**
  Ensemble against this encoder: best α=0.8 → 0.7006 (+0.0006, noise); macro 0.5874 at α=0.7
  (+0.007, a whisper of tail complementarity). The encoder is still redundant with the n-grams.

  **Conclusion for Review 3 — the statute-ID question is settled for this corpus.** The
  domain-pretrained encoder, with 5.4x data, double the epochs, a properly-fed head and full-document
  context, reaches 0.53 where a linear model over hashed 1–2-grams reaches 0.70, and adds nothing in
  ensemble. Whatever the encoder learns, the n-grams already have. The defensible position: **TF-IDF
  is the statute classifier; the encoder's role is element verification**, where reading (not
  keyword presence) is what the task needs and where a linear model cannot compete. Total encoder
  compute spent reaching this: ~3.5 GPU-hours across 6 runs, all logged in RESULTS.md.
- **IPC→BNS mapping wired in:** 69/100 rows auto-applicable, 29 held for review, 7 non-BNS targets
  routed out. Auto-apply is gated on `confidence == high AND needs_review == N AND target is a real
  BNS section`.
- **ASR baseline:** faster-whisper large-v3, **int8 confirmed best of 4 compute types** (finding 6).
  **Baseline of record: 60 clips → WER 55–57%, CER 20–22%, RTF ~1.8** across two runs (55.37/20.11 and
  57.25/22.37; the spread is run-to-run noise — GPU decode is not deterministic, see 6b). The high WER
  is real, not a bug. ⚠️ 2 of 60 clips fabricate content (replay loop, language drift); the quality
  guard catches both — finding 6b is the safety-relevant result.
- **LangGraph slice** wired: `(audio → ASR) → statute-ID → IPC/BNS → cognizability → FIR/CSR/review`,
  with a conditional entry edge that skips ASR for text input.
- **24 pytest tests pass in ~2s**, no model weights required (`make test`).

---

## ⚠️ Findings that change assumptions — READ THESE

### 1. The GPU is 8 GB, not 16 GB
`nvidia-smi` reports **RTX 5050 Laptop, 8.15 GB**. CONTEXT.md's locked decision says "single 16 GB
consumer GPU (12 GB = degraded fallback)" — the real box is below even the fallback tier. Host RAM is
15.6 GB; WSL sees ~7.7 GB.

What still fits: the whole vertical slice. InLegalBERT trains at batch 8 × 512 tokens in **4.6 GB**.
Whisper large-v3 int8 needs ~1.6–4 GB.

**What does not fit: Qwen2.5-14B 4-bit (~9 GB), the planned synthetic-data generator.** Qwen2.5-7B
4-bit (~4.7 GB) still does. Every model in `configs/models.yaml` is now annotated `fits: true|false`
against the real 8 GB budget. **This is an open decision, not resolved** — options are a smaller
data-gen model, CPU offload (slow), or generating on different hardware.

### 2. Blackwell (sm_120) needs CUDA 12.8+
cu121 wheels have no sm_120 kernels. `setup_wsl.sh` installs torch from the cu128 index explicitly
rather than letting pip resolve it.

### 3. CTranslate2 aborts at the C level when cuDNN is missing — silently
faster-whisper `dlopen`s cuDNN 9 by soname. torch's pip wheel ships it inside `site-packages/nvidia/`,
which is not on the loader path, so it fails with `Cannot load symbol cudnnCreateTensorDescriptor` and
**kills the process without raising a Python exception** — the device-fallback `try/except` in
`whisper.py` could not catch it, and the run exited 0 with no results. Fixed by preloading torch's
bundled cuBLAS + cuDNN via `ctypes.CDLL(..., RTLD_GLOBAL)` before the model loads
(`fir.asr.whisper.preload_cuda_libs`). Worth remembering: the failure mode is a *silent* exit.

### 4. FLEURS TSVs have no `speaker_id` column
Verified against the real `ta_in/test.tsv`: columns are
`id, file_name, raw_transcription, transcription, grapheme_transcription, num_samples, gender`.
Several dataset cards claim a `speaker_id` at index 5. Getting this wrong silently reported
"0.0s of audio" and a meaningless RTF.

### 5. Domain shift (risk R10) is now measured, not theoretical
The ILSI-trained classifier works on long court text and produces **nothing** on short complaint text.
Demonstrated on a single document, same model, two lengths:

| input | prediction |
|---|---|
| real ILSI doc, 9,527 chars (gold: 302, 324, 304) | predicts 302, 324, 304 + 149, 300, 147, 148 → FIR |
| **same doc truncated to 300 chars** | predicts nothing → officer review |

Short, complaint-style narratives fall below threshold because the model is calibrated on ~7,180-char
documents. **This is the strongest evidence yet that the synthetic in-domain data (PLAN §5.2) is on the
critical path, not a nice-to-have** — statute-ID accuracy on ILSI does not transfer to the actual
input the system will receive.

### 5b. The statute classifier is English-only — Tamil input needs a translation node
Running a spoken-Tamil complaint through the real pipeline (2026-09-13): extraction filled date
(`--05-08T22:00`), amount (₹50,000) and phone, all with provenance and the narrative gate passed —
but **statute-ID predicted nothing**, because ILSI is English court text and so is everything the
classifier has ever seen. Finding 5's domain shift is *language* as well as length.

DATASETS.md caveat 2(b) already said statute reasoning would run "on the extracted/translated English
narrative". **That translation node does not exist yet and belongs before `statute_id` in the graph.**
Local-only options: IndicTrans2 (AI4Bharat, ~1 GB, en↔ta) or the resident Qwen2.5-7B. Either way it
is one more thing the 8 GB budget has to hold. Until then, Tamil complaints route to the officer with
the structured slots filled and no sections — correct behaviour, but it is the officer doing Stage C.

### 5c. ⚠️ SAFETY-RELEVANT: a translation error turned a dowry-cruelty complaint into a CSR
First real end-to-end run with the interim translator (opus-mt-dra-en) on a Tamil complaint:

> "என் கணவர் மற்றும் அவரது தாயார் **வரதட்சணை** கேட்டு என்னை கொடுமைப்படுத்தினார்கள். … கொலை செய்வேன் என்று மிரட்டினார்கள்."
> → *"My husband and his mother asked for a **relief** and continued beating me. … threatened to kill me."*

"Dowry" became "relief". The classifier, reading the English, predicted only **506** (criminal
intimidation, non-cognizable) → BNS 351(2) → **route CSR**. A 498A complaint — cognizable, an FIR —
would have been logged as a Community Service Register entry. Every component did its job; the
legal outcome was still wrong, because one word did not survive translation.

**Fix — the element cue scan as a recall safety net.** The element cue lists (`elements.py`) are
bilingual and run on the *original* transcript, so they are immune to translation loss. `scan_all`
now checks every covered section's ingredients against the original; a section whose ingredients are
**all** present but which the classifier did not predict becomes a `cue_hit`. Cue hits are a
separate channel from the review queue (a plain theft also trips 380's cues — a shop is a building —
and that must not disturb a correct FIR). They **force OFFICER_REVIEW only when they would raise the
route**: nothing cognizable applied + a cognizable cue hit → the officer is asked, with the section
and the Tamil words named. Same complaint now: *"REVIEW — original narrative evidences cognizable
offence(s) the classifier did not predict: IPC 498A (கணவர், வரதட்சணை)"*. Pinned in
`tests/test_cue_scan.py` with the actual bad translation as the fixture.

While building it: 302's "death caused" element fired on கொலை in a *threat* ("I will kill you");
tightened to actual-death words.

**Translation quality, measured** (`harness/run_translation_eval.py`, 336 FLEURS/FLoRes ta→en pairs
joined on sentence id): opus-mt-dra-en **chrF 35.2, BLEU 10.5**, 0.3 s/sentence on GPU. Published
FLoRes ta→en is ~50 chrF for NLLB-600M and ~55–60 for IndicTrans2 — the interim model is ~20 points
behind, and "dowry → relief" is what that gap looks like on a legal text. **Translation quality is an
upper bound on statute-ID quality for Tamil input**; upgrading the model (Q-15 / next-open item 2)
is the single biggest lever for Tamil complaints, and the cue scan is the net under it meanwhile.

### 6. Tamil ASR is genuinely hard for Whisper — the high WER is real, not a bug
The first run scored WER 62.12% at RTF 3.467 and two hypotheses contained Hebrew characters, which
looked like degraded int8 numerics. **That hypothesis was tested and disproved.**
`harness/compare_asr_compute_types.py` (`make asr-compare`), 10 clips:

| config | WER | CER | RTF |
|---|---|---|---|
| **int8 / vad** | **54.41%** | **28.25%** | 4.196 |
| float16 / vad | 60.78% | 37.63% | 2.086 |
| float16 / no-vad | 62.25% | 37.58% | 2.064 |
| int8_float16 / vad | 55.39% | 28.42% | 3.444 |

Conclusions:
- **int8 is the best config, not the culprit.** float16 is ~2x faster but 6 WER points worse. Keep int8.
- **RTF > 2 in every config**, so the slowness is not quantisation-specific — this GPU decodes large-v3
  slower than real time regardless. Batching (`faster_whisper.BatchedInferencePipeline`) and beam_size
  are the levers, not compute type. Relevant to the live-complaint latency budget.
- The errors are **vowel-length and retroflex/dental confusions** (வீரர்கள்→விரர்கள்,
  தீர்மானிப்பதை→தீர்மணிப்பதை) — real Tamil recognition errors. Because Tamil is agglutinative, one
  such slip invalidates a whole word, which is why CER (28%) sits far below WER (54%). **Report CER
  alongside WER for Tamil; WER alone overstates the failure.**
- **20 clips is too few to benchmark on.** The same int8 config scored 62.12% on 20 clips and 54.41%
  on 10 — an 8-point swing from subset choice alone.

**BASELINE OF RECORD — 60 clips, 742.7s, large-v3 int8/vad on CUDA:**

| metric | corpus | median clip |
|---|---|---|
| WER | **55.37%** | 50.00% |
| CER | **20.11%** | 11.30% |
| RTF | 1.843 | — |

WER quartiles 0.338 / 0.500 / 0.678 — the error is **broadly distributed, not outlier-driven**. This is
the honest zero-shot floor, and it is exactly why the plan carries IndicWhisper / IndicConformer and a
code-switch fine-tune. Nischay's first real task.

### 6b. ⚠️ SAFETY-RELEVANT: Whisper fabricated content on 2 of 60 clips
Three clips scored **WER > 100%**. On inspection (2026-09-13) they were **three different things**:

| clip | length | WER | CER | what actually happened |
|---|---|---|---|---|
| 1916 | 45.9s | 226% | 204% | **Replay loop** — decoder re-emitted the first 30s window; 68 words for 23 spoken |
| 1996 | 22.0s | 126% | 59% | **Language drift** — mid-sentence slide into Cyrillic/Greek/English garbage despite `language="ta"` |
| 1731 | 8.8s | 150% | 13% | **Not a fault.** 4 ref words → 6 hyp words: agglutination boundary split (தீவுக்கூட்டங்களிலும் → தீவு கூட்டங்களிலும்). WER lies; CER tells the truth. |

So two real fabrications (3%), not three. Both put words in a complainant's mouth and feed invented
facts into a document an officer signs. **The faithfulness problem starts at Stage A.**

**Mitigated 2026-09-13 — by the guard, not the decoder flag.** `fir.asr.quality` flags both patterns
post-hoc and a flagged transcript is **never auto-resolvable** downstream. Rerun of the same 60 clips
with the guard live: **2 of 60 flagged, both correctly** (1916 replay 41% duplicated; a Cyrillic drift).

What did *not* work: `condition_on_previous_text=False`, which I had predicted would stop the replay at
source. Same 60 clips, same model:

| | cond=True | cond=False |
|---|---|---|
| WER / CER | 55.37% / 20.11% | 57.25% / 22.37% |
| clip 1916 replay | 226% | **226% — unchanged** |
| language drift | clip 1996 | clip **1681** |

The replay is not caused by prompt conditioning across windows. And the drift moved to a different clip
between two otherwise identical runs, so **Whisper's GPU decode is not run-to-run deterministic** — the
±2 WER points are noise, and per-clip failures are stochastic. That is a second argument for a
post-hoc guard over any decoder setting: the setting cannot be validated on one run. Kept at `False`
(standard advice, no measured cost). Root cause of 1916 still open — VAD chunking of the 45.9s clip and
end-of-segment hallucination are the remaining suspects; a per-config diagnostic on that one clip is
queued for when the GPU and RAM are free.

Known gap: hallucinated *English* ("eddie devastated atises of super g") passes the guard, because Latin
is legitimate in code-switched speech — catching it needs a plausibility model.

### 6d. WSL RAM (7.7 GB), not the GPU, is the binding constraint for full-corpus runs
The full-corpus TF-IDF fit was **OOM-killed at 2.6 GB RSS** when a Whisper load ran alongside it. Cause:
liblinear converts X to float64 *per worker*, ~750 MB each on 42,834 docs, × `n_jobs=6`. Now `n_jobs=2`
and a float32 vectorizer. **Rule: one memory-heavy job at a time on this box.** The proper fix is
`%USERPROFILE%\.wslconfig` with `memory=12GB` (host has 15.6 GB) — not applied; it is a persistent
config change and needs `wsl --shutdown`.

### 6e. ⚠️ The WSL VM restarts itself under the full-corpus vectorization — root cause unknown
Five consecutive full-corpus attempts died at the same point (vectorizing 42,835 docs, ~1.6–2.3 GB
RSS, ~95 s in). Only the first was a Linux OOM (`dmesg`, 2.6 GB, alongside a Whisper load). The
other four left **no `dmesg` entry, no Python traceback, no fault-handler output, and killed the
whole `wsl` session** — and `uptime` reset each time. The **VM itself rebooted** (Hyper-V VmSwitch
events in the Windows log confirm network teardown/re-creation at each death).

Ruled out by direct test, each surviving cleanly: a 3.5 GB allocation; 150 s of pure CPU; a 400 s
background session with foreground `wsl` calls landing on it; 75 s of idle. No `.wslconfig`, no
cgroup limit, no Docker, no crash dump written, host commit fine (36.6 GB limit, 17 GB free).
The kernel cmdline has `panic=-1`, so a guest panic would look exactly like this, but the dump
folder is empty.

**Diagnosis narrowed (later the same day, after a 6th death — BERT tokenization of 42k docs):**

| workload | allocation pattern | VM |
|---|---|---|
| 3.5 GB numpy allocation, touched | few large contiguous blocks | survived |
| 150 s pure CPU | none | survived |
| TfidfVectorizer vocabulary, 42k docs | millions of small dict entries | **died** |
| HashingVectorizer, one call, 42k docs | large nested Python lists → CSR | **died** |
| HashingVectorizer, 8k-doc chunks + vstack | small lists, freed per chunk | **survived** |
| tokenizer, one call, 42k docs (~22M Python ints) | nested lists → tensor | **died** |

**Rule: a process whose *Python heap* (many small objects) grows past ~2 GB takes the VM down;
numpy/torch block allocations of the same size do not.** Mechanism still unproven (page-table churn
vs. WSL's memory-reclaim is the leading guess) but the predictor is reliable and the fix is
mechanical: **chunk every corpus-scale Python-object build** and convert to arrays per chunk. Applied
to the vectorizer (8k docs), the tokenizer (2k docs), and `TfidfStatuteClassifier.predict_proba`
(4k docs). Full-corpus TF-IDF then completed (finding 8 table above).

Also part of what got through: launching via `Start-Process wsl.exe` (detached from the tool) and
monitoring only from Windows. Whether those mattered independently is untested.

Operational rules that each cost a run:
- **One memory-heavy job at a time on this box.** The one confirmed OOM was Whisper + TF-IDF together.
- **`nohup ... &` inside a `wsl -e bash -lc` session does not survive the session ending.** WSL
  kills the process tree. Long jobs need a `wsl` invocation that stays attached, or `Start-Process`.
- **`/tmp` is wiped on every VM restart.** Logs go to `artifacts/logs/` on `/mnt/c` (gitignored).
- **Proper fix to try first when resuming:** `%USERPROFILE%\.wslconfig` with `memory=12GB` and
  `swap=4GB`, then `wsl --shutdown`. Not applied — persistent config change.

### 6f. The ASR was 3-4x slower than real time because of the temperature-fallback ladder
Profiling (`harness/profile_asr.py`, `artifacts/reports/asr_profile.json`) on 8 FLEURS clips, CTranslate2
4.8.2: **RTF 3.06 with faster-whisper's default decoding vs 0.57 with one pass at temperature 0** (beam 1;
3.20 vs 0.65 at beam 5), WER equal or better. The default is a *fallback ladder* [0, .2, .4, .6, .8, 1.0]:
any window failing the compression-ratio / avg-logprob (-1.0) check is decoded again at the next
temperature, up to six times, and low-confidence Tamil fails that check constantly. VAD was not the cause
(3.40 vs 3.32); the CTranslate2 version was not the cause (4.5.0 → 4.8.2 changed nothing; 4.6.2 disables
INT8 tensor cores on Blackwell, which is why int8 is no longer faster than float16 here — float16 RTF 0.44
vs int8 0.57, same WER; int8 kept for VRAM).

On the same clips joined into one 148 s complaint-length recording, **BatchedInferencePipeline over VAD
chunks beat the 30 s sliding window on both axes: RTF 0.31 → 0.10, WER 81% → 49%** (beam 5) — chunks
follow the speaker's pauses instead of cutting sentences at fixed offsets.

Both are now the pipeline defaults (`asr.temperature: 0.0`, `asr.batch_size: 8`). **Re-baselined on the 60
clips: WER 57.25% → 51.41%, CER 22.37% → 15.05%, RTF 1.81 → 0.46** (sequential T=0 in between: 53.49% /
15.82% / 0.51; `asr_baseline_seq.json`; the old run is kept as `asr_baseline_fallback_ladder.json`).
**What did not change: the repetition guard still trips on 2/60 clips (1916 and 1721) at T=0**, so the
ladder explains the speed, not the replay loop — the hypothesis that fabrications came from the sampled
fallback passes is *not* supported for the replay case (the Cyrillic drift did not recur, but 2 clips is
not evidence). The guard stays; next lever is `no_repeat_ngram_size` / repetition penalty in decoding.
The earlier compute-type sweep (int8 best, float16 worse) was run under the ladder and is superseded.

### 6c. For Tamil, WER > 100% does not mean fabrication — report CER alongside
Clip 1731 is the cautionary case: the transcript was essentially right and WER called it the second-worst
in the set. Tamil's agglutination means a single boundary decision changes the word count. **Any Tamil
ASR result must show CER next to WER**, and per-clip WER > 100% needs a CER check before it is called a
hallucination.

FLEURS also records each sentence with several speakers under the **same clip ID**. Clip 1996's other
speaker decoded cleanly (WER 52%). The guard judges the decode, not the sentence — and fixture
extraction by ID alone silently picks the wrong take.

### 7. The InLegalBERT head collapsed to all-zeros — plain BCE is the wrong loss here
First InLegalBERT run: test micro-F1 **0.2394**, macro-F1 **0.0175**, and — the diagnostic number —
**6 of 100 labels ever predicted** (TF-IDF predicts 99). Training loss flatlined at 0.142 for the whole
of epoch 2 and dev micro-F1 moved 0.2521 → 0.2522 between epochs. That is not underfitting; it is
converging to the trivial solution.

Cause: ILSI averages **3.67 positive labels out of 100**, so unweighted `BCEWithLogitsLoss` is
minimised by predicting all-zeros. The comparison was also **unfair by construction** — the TF-IDF
baseline was given `class_weight="balanced"` while the encoder got plain BCE.

Fix applied: per-label `pos_weight = neg/pos`, clamped at 50 so a 1-example label cannot destabilise
training (`configs/pipeline.yaml: pos_weight`). Note the clamp binds for the **median** label, which
says how long-tailed ILSI is: over half the sections appear in fewer than 157 of 8,000 documents.

`pos_weight` fixed the collapse but did not close the gap to TF-IDF — see finding 8, which tested the
obvious explanation (truncation) and ruled it out.

### 9. The hand-coded cognizability stub was wrong on 4 of 48 entries
Parsing the official First Schedule (`make schedule --check`) against the stub that had carried the
slice since 2026-08-20: **BNS 126 (wrongful restraint), 223 (disobedience to a public servant's
order), 296 (obscene acts) and 329 (criminal trespass) are cognizable in the Schedule; the stub said
non-cognizable.** All four are refusals-to-register errors — the direction that is misconduct — and
the stub was written from memory of the IPC-era classification "for common offences". Three more
stub entries (61, 85, 190, 238) were flat where the Schedule is conditional, and 303 (theft) hides a
value split the stub did not know about. Lesson for the write-up: a cognizability table is legal
authority, not a config file; it must come from the gazette text with page citations, and it did
not until today. The CSV still needs a legal read-through (`bnss_schedule1_rows.csv` is laid out for
that) — parsing the official text is necessary, not sufficient.

Two Schedule facts that change pipeline behaviour: (a) **theft (303(2)) is non-cognizable below
Rs 5,000**, so the extracted property value now decides the route and a complaint with no value
asks the officer; (b) **s.85 cruelty is cognizable only when reported by the aggrieved woman, a
relative, or a notified public servant**, which the pipeline cannot verify — so every 498A case is
officer review with that condition quoted, not an automatic FIR as the stub had it.

### 8. Truncation is NOT why the encoder loses — tested and ruled out
The obvious explanation for finding 7's gap was the 512-token window on ~7,180-char documents. It was
tested directly (`harness/ablate_truncation.py`): train the *same* TF-IDF model twice, once on full
documents and once on documents cut to the exact 512 wordpieces InLegalBERT receives, using
InLegalBERT's own tokenizer so the budget is identical rather than approximated.

| TF-IDF input | micro-F1 | macro-F1 | labels used |
|---|---|---|---|
| full document | 0.5988 | 0.4278 | 99 |
| truncated to 512 wordpieces | **0.5472** | 0.3924 | 99 |
| *(InLegalBERT, same 512 budget)* | *0.2846* | *0.1811* | *79* |

**Truncation costs only 0.0516 micro-F1 (8.6% relative) — about 16% of the 0.314 gap.** Given the same
input, the bag-of-words model still beats the encoder nearly 2:1. So the context window is a real but
minor effect, and hierarchical chunking is *not* the highest-value next step.

The likely causes, in order to try:
1. **Undertrained.** Effective batch 16 × 2 epochs over 8,000 docs = only ~1,000 optimizer steps for a
   randomly-initialised 100-way head. TF-IDF with 200k n-gram features is very strong in this regime.
2. **Data starved.** The slice trains on 8,000 of 42,834 available docs (19%). Linear models tolerate
   small data; encoders do not. **`make slice-full` is the single highest-value experiment** — 5.4x the
   data should help the encoder far more than the baseline.
3. **Uniform pos_weight.** The clamp binds for every label (median = max = 50), so per-label balancing
   is effectively lost; the tuned threshold drifted to 0.65. Try a lower clamp or focal loss.
4. Separate, higher LR for the freshly-initialised classifier head.

**Broader lesson worth carrying into Review 3:** on this task, a cheap sparse baseline beats a
domain-pretrained legal encoder by 2x. Keep TF-IDF in every comparison as the honest floor — a paper
result claiming an encoder win without this baseline would not be trustworthy.

---

## Review 3 (panel, 16 Sep 2026) — deliverables built 2026-09-13
Guidelines: `C:\Users\madha\Downloads\BCSE497J_Project_I_Guidelines.pdf` (Review 3 = 20 marks, 7 parameters).
- `Review3_Presentation.pptx/.pdf` — 18 slides, rubric map on slide 2; built by
  `scripts/review3/build_deck.py` from the JSON reports (rebuild after any new result, then export from
  PowerPoint). Figures: `scripts/review3/make_figures.py` → `assets/r3_*.png`.
- `docs/REVIEW3_DEMO.md` — demo script (6 steps, demo-mode URLs `/?sample=theft&run=1`), fallbacks,
  likely panel questions with one-line answers, and the **things only the team can fill in**: the Review 2
  panel comments → action table (slide 3), the individual-contribution table (slide 16), the AI-tool
  acknowledgement wording, a publication venue (Review 4 gives 2 marks only against proof of submission).
- `Review3_Report.docx/.pdf` was also generated (Chapters 1–3 + §3.8 revisions + Chapter 4 draft) before
  Madhav said no report is asked for at Review 3 — left in place as the seed for the Review 4 report
  (`scripts/review3/build_report.py`); not maintained further.

## In flight at end of session (2026-09-13)
**Nothing.** All GPU queues completed (`queue1_done.log`, `queue.log`, `asr_queue.log`). All results are
in `artifacts/reports/*.json` and rendered in `RESULTS.md` (`make results`). The GPU is free.
Housekeeping done at the end: ruff clean, repo normalised to LF (`.gitattributes`; 38 files had drifted
to CRLF via Windows-side edits), `scripts/run.sh` stands in for the missing `make`. CTranslate2 upgraded
4.5.0 → 4.8.2 (CUDA 12.8 / Blackwell build); pdfplumber added for the gazette parsers.

## Next / open
1. **Statute-ID: settled — TF-IDF stays the classifier** (see the "every lever at once" result under
   finding 8). Do not spend more GPU on encoder *classification*. The encoder's next job is the
   **element verifier** (PLAN P3): replace the keyword cues in `elements.py` with a model that reads
   the narrative against each element's statute text. That is also where Qwen2.5-7B could serve
   (finding 1 / Q-15) instead of a fine-tuned encoder.
2. **Translation node — built with an interim model; upgrade it.** The node exists and runs with
   opus-mt-dra-en. To get IndicTrans2 (much better): `huggingface-cli login`, accept the terms at
   huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M, set `translate.model` in pipeline.yaml,
   run `scripts/download_indictrans.py`. Alternatively approve the 2.5 GB NLLB-600M download.
3. **Decide the data-gen model** given 8 GB (finding 1) — blocks PLAN §5.2. Note 4-bit was always the
   plan and does not rescue the 14B: Q4_K_M *is* the ~9 GB figure, vs 7.1 GB usable. Full table and
   throughput stakes in OPEN-QUESTIONS.md **Q-15**. **Deferred 2026-09-10 (laptop on battery)** —
   when next on mains power, install Ollama and benchmark 14B-Q4-with-offload vs 7B-Q4 on a real
   Tamil complaint→IF-1 prompt. Must be plugged in: laptop GPUs downclock on battery.
4. **ASR**: decoding fixed (finding 6f: T=0 + batched VAD chunks; WER 51.41% / CER 15.05% / RTF 0.46).
   **Still open: the replay loop on 2/60 clips** — try `no_repeat_ngram_size` / `repetition_penalty` in
   `WhisperAsr`, measure on the 60 clips, keep the guard. Then benchmark IndicWhisper / IndicConformer
   (~1.5 GB download — ask) against 51.41% / 15.05%. int8 vs float16: same WER without the ladder,
   float16 faster (RTF 0.44 vs 0.57 on 8 clips); int8 kept for VRAM until the LLM co-residence is known.
5. Finish the mapping table: verify the 29 `needs_review` rows and fill the 4 `VERIFY` rows
   (155, 156, 190, 482) against the official MHA gazette; get legal sign-off.
6. **BNSS First Schedule: built (finding 9), needs a legal read-through.** Read
   `data/statutes/bnss_schedule1_rows.csv` against the printed Schedule (page column given); 26
   conditional entries are quoted verbatim. BNS section texts are parsed too (`bns_2023.jsonl`);
   Part II of the Schedule (offences under other laws, by punishment) only if ever needed.
7. **Classifier abstains on short complaints — and lowering the threshold would not help.** The TF-IDF
   model predicts nothing for the two-sentence "theft" sample (trained on ILSI court fact-statements —
   finding 5); the cue scan carries the case. Measured: its top-5 for that complaint is 420 (0.22), 149,
   506, 323, 302 — **379 is not in the top five**, so a lower threshold would apply *wrong* sections.
   The fix is in-domain data (synthetic complaints, PLAN §5.2), not calibration. The UI now says "no
   section reached the threshold (best score 0.22)" without naming the runner-up.
8. Then PLAN.md phases: synthetic data-gen → extraction → element verifier → instantiation → UI.

## Team split (3 people, discussed 2026-08-20)
- **Madhav** — Stage C/D: statute identification + instantiation (the research core).
- **Nischay** — Stage A: ASR. Owns finding 6, the model benchmark (Whisper vs IndicWhisper vs
  IndicConformer), Tamil normalization + **ITN** (dates/amounts/times — WER hides all of it), and the
  later code-switch LoRA.
- **Rohit** — Stage B: extraction + the synthetic data engine (IF-1 slot schema, local-LLM generator,
  InLegalNER seeding, MuRIL NER). Biggest "build our own data" effort; everything downstream waits on it.
- **Week-1 no-GPU tasks** (only one 8 GB GPU, so only one person can train at a time): the 29
  `needs_review` mapping rows, and the BNSS First Schedule cognizability table. Both are pure legal
  research and both are on the critical path.
- **Assign the eval harness explicitly** — IMPLEMENTATION.md §6 says it runs "in parallel from day 1"
  and it is the thing that otherwise becomes nobody's job.

---

## Done previously (planning + Review 2)
- Locked 4 decisions (16 GB GPU / 12 GB fallback — **now contradicted by hardware, see finding 1**;
  strict local-only incl. synthetic data-gen; synthetic-first eval; folder = Downloads\fir-tn-capstone).
- Wrote PLAN.md, ARCHITECTURE.md, OPEN-QUESTIONS.md; context bridge CONTEXT.md + CLAUDE.md.
- VERIFIED DATASETS LIVE (2026-08-17): FLEURS-ta, Kathbath, ILSI, OpenNyAI InLegalNER, BNS/BNSS text.
  Samples in `data_samples/`. Wrote DATASETS.md + IMPLEMENTATION.md.
- IPC→BNS mapping built + verified against a BNS-IPC pocket directory → ~71/100 high confidence;
  ~29 med/low flagged. **That PDF is a third-party directory, NOT the official gazette.**
- Review 2 deliverables COMPLETE (2026-08-19): Review2_Presentation.pptx/.pdf (14 slides, PRIMARY
  graded artifact), Literature_Review.docx/.pdf, Review2_Report.docx/.pdf.
  Names: 23BAI1088 Madhav K, 23BAI1245 Nischay Kuchibotla, 23BAI1416 Rohit A. S.; Guide Dr. Shivaranjani.
  NOTE: guide must APPROVE the PPT in the VIT portal before the panel review.
- PDF conversion on this Windows box uses Word/PowerPoint COM (LibreOffice/soffice.py are Unix-only).

## How to resume in a NEW session
1. Open Claude Code with this folder as the working directory — `CLAUDE.md` auto-loads `CONTEXT.md`.
2. Say: "Read PROGRESS.md and continue from 'Next / open'."
3. Everything Python runs in WSL2:
   `wsl -d Ubuntu -e bash -lc "cd /mnt/c/Users/madha/Downloads/fir-tn-capstone && source ~/.venvs/fir-tn/bin/activate && <cmd>"`
4. `make test` (2s) · `make lint` · `make slice` · `make slice-full`. **`make` is not installed in this
   WSL** (needs sudo) — `bash scripts/run.sh <target>` runs the same commands.
