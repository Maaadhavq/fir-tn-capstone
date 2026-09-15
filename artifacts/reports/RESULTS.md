# FIR-TN — results as of 2026-09-13

All numbers below are read from `artifacts/reports/*.json`; nothing is recomputed here. Caveats and interpretation live in `PROGRESS.md`.

## Statute identification (ILSI, 100 IPC sections)

| train set | n_train | n_test | model | micro-F1 | macro-F1 | micro-P | micro-R | labels used | thr |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 8k subsample | 8,000 | 2,000 | tfidf+ovr-logistic | **0.5988** | **0.4278** | 0.6569 | 0.5502 | 99/100 | 0.50 |
| 8k subsample | 8,000 | 2,000 | InLegalBERT | **0.2846** | **0.1811** | 0.2523 | 0.3264 | 79/100 | 0.65 |
| 8k subsample | 8,000 | 2,000 | InLegalBERT[head_tail] | **0.3170** | **0.2049** | 0.2789 | 0.3672 | 83/100 | 0.65 |
| 8k subsample | 8,000 | 2,000 | InLegalBERT[headlr10] | **0.3530** | **0.2398** | 0.3068 | 0.4156 | 89/100 | 0.75 |
| full corpus | 42,835 | 13,039 | tfidf+ovr-logistic | **0.7000** | **0.5805** | 0.7318 | 0.6709 | 100/100 | 0.55 |
| full corpus | 42,835 | 13,039 | InLegalBERT | **0.4636** | **0.3984** | 0.4642 | 0.4630 | 97/100 | 0.80 |
| full corpus | 42,835 | 13,039 | InLegalBERT[e4_ht_hl10] | **0.5256** | **0.4594** | 0.5168 | 0.5347 | 100/100 | 0.85 |

Threshold tuned on dev for micro-F1. `labels used` = sections receiving at least one prediction on test; a low count means collapse onto head labels.

### Truncation ablation (TF-IDF, identical model, two input budgets)

| input | micro-F1 | macro-F1 |
|---|---:|---:|
| full document | 0.5988 | 0.4278 |
| truncated to 512 wordpieces | 0.5472 | 0.3924 |

Truncation keeps 24.1% of the text and costs 0.0516 micro-F1 (8.6% relative). It does not explain the encoder gap.

### Ensemble sweep — full corpus (p = α·TF-IDF + (1−α)·BERT)

TF-IDF `hashing`, BERT `head`; n_dev=10,200, n_test=13,039. α and threshold chosen on dev.

| α | dev micro-F1 | test micro-F1 | test macro-F1 | thr |
|---:|---:|---:|---:|---:|
| 0.0 | 0.4716 | 0.4636 | 0.3984 | 0.80 |
| 0.1 | 0.5328 | 0.5244 | 0.4507 | 0.80 |
| 0.2 | 0.5918 | 0.5890 | 0.5040 | 0.75 |
| 0.3 | 0.6368 | 0.6326 | 0.5367 | 0.70 |
| 0.4 | 0.6665 | 0.6618 | 0.5596 | 0.65 |
| 0.5 | 0.6820 | 0.6785 | 0.5695 | 0.65 |
| 0.6 | 0.6950 | 0.6922 | 0.5840 | 0.60 |
| 0.7 | 0.6996 | 0.6978 | 0.5835 | 0.60 |
| 0.8 | 0.7010 | 0.6985 | 0.5801 | 0.60 |
| 0.9 | 0.7015 | 0.6986 | 0.5778 | 0.60 **←** |
| 1.0 | 0.7010 | 0.7000 | 0.5805 | 0.55 |

Gain over the best single model on test: **-0.0014** micro-F1.

### Ensemble sweep — full corpus, BERT[e4_ht_hl10] (p = α·TF-IDF + (1−α)·BERT)

TF-IDF `hashing`, BERT `head_tail`; n_dev=10,200, n_test=13,039. α and threshold chosen on dev.

| α | dev micro-F1 | test micro-F1 | test macro-F1 | thr |
|---:|---:|---:|---:|---:|
| 0.0 | 0.5341 | 0.5256 | 0.4594 | 0.85 |
| 0.1 | 0.5847 | 0.5776 | 0.5129 | 0.85 |
| 0.2 | 0.6305 | 0.6233 | 0.5462 | 0.80 |
| 0.3 | 0.6581 | 0.6525 | 0.5635 | 0.75 |
| 0.4 | 0.6761 | 0.6707 | 0.5718 | 0.70 |
| 0.5 | 0.6881 | 0.6834 | 0.5802 | 0.65 |
| 0.6 | 0.6956 | 0.6901 | 0.5794 | 0.65 |
| 0.7 | 0.7022 | 0.6993 | 0.5874 | 0.60 |
| 0.8 | 0.7035 | 0.7006 | 0.5839 | 0.60 **←** |
| 0.9 | 0.7030 | 0.7016 | 0.5858 | 0.55 |
| 1.0 | 0.7010 | 0.7000 | 0.5805 | 0.55 |

Gain over the best single model on test: **+0.0006** micro-F1.

## ASR (faster-whisper large-v3, FLEURS ta_in test)

| setting | clips | audio | device | decoding | WER | CER | median clip WER | RTF | flagged |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|
| current default | 60 | 743s | cuda/int8 | temp 0.0, cond_prev=off | **51.41%** | **15.05%** | 45.45% | 0.46 | 2 |
| temperature fallback ladder [0..1.0] | 60 | 743s | cuda/int8 | temp ladder (library default), cond_prev=off | **57.25%** | **22.37%** | 50.00% | 1.81 | 2 |
| condition_on_previous_text=True | 60 | 743s | cuda/int8 | temp ladder (library default), cond_prev=on | **55.37%** | **20.11%** | 50.00% | 1.84 | — |

Quality-guard flags (transcript may contain words not spoken):

| clip | length | WER | CER | flag |
|---|---:|---:|---:|---|
| 1916 | 45.9s | 178.26% | 132.50% | repetition: 38% of transcript duplicated |
| 1721 | 20.7s | 141.67% | 105.32% | repetition: 90% of transcript duplicated |

Tamil is agglutinative: one boundary decision changes the word count, so **CER is the honest metric** and per-clip WER > 100% is not by itself a hallucination.

### Compute-type sweep (10 clips)

| config | WER | CER | RTF |
|---|---:|---:|---:|
| int8/vad | 54.41% | 28.25% | 4.196 |
| float16/vad | 60.78% | 37.63% | 2.086 |
| float16/no-vad | 62.25% | 37.58% | 2.064 |
| int8_float16/vad | 55.39% | 28.42% | 3.444 |

Run with the library's temperature-fallback ladder on, so every RTF here carries the re-decoding cost (see the profile below); the WER ordering (int8 best) stands.

### Where the time goes (8 clips; NVIDIA GeForce RTX 5050 Laptop GPU; CTranslate2 4.8.2)

| compute | mode | beam | temp. fallback | batch | audio | RTF | WER |
|---|---|---:|---|---:|---:|---:|---:|
| int8 | per-clip | 1 | on | — | 144s | 3.057 | 65.83% |
| int8 | per-clip | 1 | off | — | 144s | 0.574 | 59.95% |
| int8 | per-clip | 5 | on | — | 144s | 3.199 | 60.92% |
| int8 | per-clip | 5 | off | — | 144s | 0.653 | 59.31% |
| int8 | joined-sequential | 1 | off | — | 148s | 0.252 | 82.56% |
| int8 | joined-batched | 1 | off | 8 | 148s | 0.080 | 65.70% |
| int8 | joined-batched | 1 | off | 16 | 148s | 0.083 | 65.70% |
| int8 | joined-sequential | 5 | off | — | 148s | 0.310 | 80.81% |
| int8 | joined-batched | 5 | off | 8 | 148s | 0.105 | 49.42% |
| int8 | joined-batched | 5 | off | 16 | 148s | 0.104 | 49.42% |
| float16 | per-clip | 1 | on | — | 144s | 2.210 | 63.65% |
| float16 | per-clip | 1 | off | — | 144s | 0.439 | 59.99% |
| float16 | per-clip | 5 | on | — | 144s | 2.496 | 61.39% |
| float16 | per-clip | 5 | off | — | 144s | 0.504 | 55.21% |
| float16 | joined-sequential | 1 | off | — | 148s | 0.196 | 82.56% |
| float16 | joined-batched | 1 | off | 8 | 148s | 0.067 | 59.30% |
| float16 | joined-batched | 1 | off | 16 | 148s | 0.069 | 59.30% |
| float16 | joined-sequential | 5 | off | — | 148s | 0.238 | 80.81% |
| float16 | joined-batched | 5 | off | 8 | 148s | 0.097 | 51.74% |
| float16 | joined-batched | 5 | off | 16 | 148s | 0.099 | 51.74% |

`joined-*` rows transcribe the clips concatenated into one complaint-length recording; `batched` uses faster-whisper's BatchedInferencePipeline over its VAD chunks. Compare within one run only (laptop GPU clocks vary).

## Translation ta→en (feeds the English-only statute classifier)

Gold: FLEURS ta_in/test x en_us/test joined on FLoRes sentence id. FLoRes is Wikipedia register, not complaints — a model-comparison signal, not a deployment estimate.

| model | backend | n | chrF | BLEU | s/sentence | device |
|---|---|---:|---:|---:|---:|---|
| `Helsinki-NLP/opus-mt-dra-en` | marian | 336 | **35.2** | 10.5 | 0.30 | cuda |

For scale: published FLoRes ta→en is ~50 chrF (NLLB-600M) and ~55–60 (IndicTrans2). Translation quality bounds statute-ID quality for Tamil input; the element cue scan on the original text is the safety net under it (PROGRESS.md finding 5c).

## Extraction (rule-based floor, per-slot)

Gold: `extraction_gold.jsonl`, 28 hand-authored adversarial documents (smoke set, not a benchmark).

| slot | tp | fp | fn | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|
| date | 18 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| time | 16 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| amount_inr | 17 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| phone | 7 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| vehicle | 5 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| **micro** | 63 | 0 | 0 | 1.000 | 1.000 | 1.000 |

Zero false positives is the property that matters: a wrong value on the form invites trust; a blank prompts a question.

## IPC → BNS mapping

| rows | auto-applicable | held for review | non-BNS targets |
|---:|---:|---:|---:|
| 100 | 69 | 29 | 7 |

Auto-apply rule: `confidence == high AND needs_review == N AND target is a BNS section`. Source is a third-party pocket directory, not the MHA gazette.
