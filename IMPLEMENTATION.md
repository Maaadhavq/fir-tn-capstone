# IMPLEMENTATION.md — Coding Plan (grounded in verified data)

> Companion to PLAN.md (roadmap) and DATASETS.md (verified data). This file is the **engineering
> plan**: environment, repo layout, the data layer, and a concrete first vertical slice that is
> provably buildable **now** with the confirmed open datasets.

## 0. Doability at a glance

| Component | Buildable now? | With what data |
|-----------|----------------|----------------|
| Statute-ID baseline (multi-label, IPC) | ✅ yes, week 1 | ILSI (downloaded, samples shown) |
| ASR baseline + WER | ✅ yes, week 1 | FLEURS-ta (2.76 GB, confirmed) |
| IPC→BNS mapping stub | ✅ yes, week 1 | MHA table (seed the common ~50 sections) |
| Legal NER seed | ✅ yes, week 2 | OpenNyAI InLegalNER |
| IF-1 extraction | ⚠️ after synthetic-gen | local LLM synth (Qwen2.5-14B 4-bit) |
| Narrative + faithfulness | ⚠️ after extraction | synthetic pairs |

## 1. Environment (Windows 11 + single 16 GB GPU)

**Critical machine constraint:** Smart App Control on this box blocks `pyarrow`, and HF `datasets`,
`bitsandbytes`, and much of the training stack depend on it. Two clean options:

- **Recommended — WSL2 (Ubuntu) or Docker for all data/training work.** A Linux container sidesteps the
  pyarrow block entirely and is where `bitsandbytes` (QLoRA), `faster-whisper`, and `vLLM` run
  smoothly. GPU passes through via CUDA-on-WSL / `--gpus all`. Still 100% local — no hosted APIs.
- **Native-Windows fallback (inference/serving only).** Avoid `datasets`; use `huggingface_hub`
  `snapshot_download` for files and **DuckDB** to read parquet (`duckdb.read_parquet(...)`). Run the
  LLM via **Ollama** (native Windows, no pyarrow). Good enough for the FastAPI serving path.

Core stack (pin in `environment/`):
```
python 3.11 · torch (CUDA 12.x) · faster-whisper (CTranslate2 int8) · transformers · peft ·
bitsandbytes (Linux) · accelerate · sentence-transformers / FlagEmbedding (BGE-m3) · spacy ·
scikit-learn · seqeval · jiwer · duckdb · huggingface_hub · fastapi · uvicorn · pydantic v2 ·
langgraph · psycopg[binary] · dvc · mlflow (offline)
LLM serving: Ollama (Qwen2.5-7B-Instruct-q4) — resident; Qwen2.5-14B-q4 for offline data-gen.
```

## 2. Repo layout (monorepo, `fir-tn/`)

```
fir-tn/
├── environment/           # dockerfiles, wsl setup, requirements, launch.json
├── data/                  # DVC-tracked; raw datasets NOT committed
│   ├── ilsi/  fleurs/  kathbath/  opennyai_ner/  statutes/  mapping/  synthetic/
├── configs/               # yaml: datasets.yaml, models.yaml, pipeline.yaml
├── src/fir/
│   ├── schema/            # FIR JSON Schema + Pydantic models (from ARCHITECTURE.md §5)
│   ├── data/              # loaders (one per dataset) + registry
│   ├── asr/               # faster-whisper wrapper, normalization/ITN
│   ├── extract/           # A: encoder-NER  |  B: LLM structured (schema-constrained)
│   ├── statute/           # classifier, RAG retriever, element verifier, ipc_bns_map, cognizability
│   ├── instantiate/       # template engine, narrative composer, faithfulness gate, pdf
│   ├── orchestrator/      # LangGraph graph wiring A→B→C→D
│   └── serving/           # FastAPI app
├── harness/               # eval: metrics per stage, config-driven runs, MLflow
├── tests/                 # pytest incl. golden-regression on a tiny fixture set
└── frontend/              # React officer UI (later)
```

## 3. Data layer (the first thing to build)

A small registry + one loader per dataset, all returning normalized Python objects — no pyarrow.

`configs/datasets.yaml`
```yaml
ilsi:
  source: zenodo
  record: "6053791"
  files: [secs.jsonl, label_vocab.json, train.jsonl, dev.jsonl, test.jsonl, type_map.json]
fleurs_ta:  {source: hf, repo: google/fleurs, config: ta_in}
kathbath_ta:{source: hf, repo: ai4bharat/Kathbath, filter: tamil}
inlegalner: {source: git, url: https://github.com/Legal-NLP-EkStep/legal_NER}
statutes:   {source: local, files: [bns_2023.jsonl, bnss_2023.jsonl, bnss_schedule1.csv]}
ipc_bns_map:{source: local, files: [ipc_bns_map.csv]}
```

`src/fir/data/ilsi.py` (sketch — reads the exact format we verified)
```python
def load_ilsi(split: str) -> list[FactInstance]:
    # each line: {"id", "text": [sentences...], "labels": ["Section N in The Indian Penal Code", ...]}
    rows = [json.loads(l) for l in open(f"data/ilsi/{split}.jsonl", encoding="utf-8")]
    return [FactInstance(id=r["id"], text=" ".join(r["text"]),
                         ipc_labels=r["labels"]) for r in rows]

def load_ipc_bns_map() -> dict[str, list[str]]:
    # {"Section 302 in The Indian Penal Code": ["BNS 103"], "498A": ["BNS 85"], ...}
    ...
```

## 4. First vertical slice — "prove it end to end" (weeks 1–2)

Goal: a thin, real pipeline on confirmed data, so doability is demonstrated before scaling.

1. **Statute-ID baseline (ILSI, IPC).** Load ILSI → TF-IDF + one-vs-rest logistic (fast baseline), then
   InLegalBERT multi-label head. Report **micro/macro-F1** on `test.jsonl`. This reproduces a
   LeSICiN-comparable baseline and is 100% doable today.
2. **IPC→BNS mapping applied.** Seed `ipc_bns_map.csv` with the ~50 most common sections (incl. the
   302→103, 304B→80, 498A→85, 34→3(5) set shown in DATASETS.md). Emit BNS sections from IPC predictions.
3. **ASR baseline (FLEURS-ta).** Run faster-whisper large-v3 (int8) on 20 FLEURS-ta clips → **WER/CER**
   via jiwer. Confirms the 16 GB budget and the ASR path.
4. **Cognizability stub.** Hard-code BNSS First-Schedule class for the seeded sections; route
   cognizable→FIR / non-cognizable→CSR.
5. **Wire with LangGraph** into a toy graph: (text → statute-ID → map → cognizability) and
   (audio → ASR → WER). One `pytest` golden test pins outputs on a 5-example fixture.

Exit: `make slice` runs both mini-pipelines and prints an F1 table + a WER number. That is the
"doable" proof, built only from data we verified this session.

## 5. Then scale (maps to PLAN.md phases)

- P0.2/P0.3 harden schema + legal KB; parse BNS/BNSS text for RAG (BGE-m3 index).
- P1 full ASR (code-switch FT, ITN). P2 both extraction approaches on **synthetic** IF-1 data.
- P3 element-wise verifier + faithful justifications (the research core).
- P4 instantiation + faithfulness gate. P5 officer UI. P6 human gold + evaluation.

## 6. Build order (dependency-safe)

`schema → data layer → statute-ID slice + ASR slice → synthetic data-gen → extraction → statute
verifier → instantiation → orchestrator → serving → UI → eval harness (in parallel from day 1)`.
