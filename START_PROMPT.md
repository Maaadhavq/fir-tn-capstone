# Starter prompt — begin implementation

Open a **new Claude Code session with `Downloads\fir-tn-capstone\` as the working directory**
(the `CLAUDE.md` there auto-loads `CONTEXT.md`), then paste the block below.

---

We are starting **implementation** of the speech-driven FIR drafting capstone. Planning and the
Review-2 deliverables are done. First read `CONTEXT.md`, `PROGRESS.md`, `IMPLEMENTATION.md`,
`DATASETS.md`, and `data/mapping/README.md` so you have the full picture.

**Hard constraints**
- Strict **local-only** (no hosted APIs), single **16 GB** consumer GPU, PEFT (LoRA/QLoRA) + quantized inference.
- This Windows machine blocks `pyarrow` (Smart App Control), and HF `datasets`/`bitsandbytes` depend on it.
  So do the data/training work inside **WSL2 (Ubuntu) or Docker** (GPU passthrough). If staying native, use
  **DuckDB + `huggingface_hub`** and run any LLM via **Ollama**.
- Decision-support only: the officer is the author of record; nothing is filed automatically.

**Task 1 — project skeleton + environment.**
Create the monorepo layout from `IMPLEMENTATION.md` §2 (`src/fir/…`, `data/`, `configs/`, `harness/`,
`tests/`) and a Python 3.11 environment (WSL2/Docker) with: torch (CUDA), faster-whisper, transformers,
scikit-learn, seqeval, jiwer, duckdb, huggingface_hub. Add a `configs/datasets.yaml` registry.

**Task 2 — first vertical slice** (`IMPLEMENTATION.md` §4 — provably doable with data we've verified):
1. **Statute-ID baseline on ILSI.** Download ILSI from Zenodo (`10.5281/zenodo.6053791`; sample rows already
   in `data_samples/`). Load `train/dev/test.jsonl` (fact text + IPC labels). Train a TF-IDF + one-vs-rest
   logistic baseline, then an InLegalBERT multi-label head. Report **micro/macro-F1** on the test split.
2. **Apply the IPC→BNS mapping.** Use `data/mapping/ipc_bns_map.csv` to convert predicted IPC sections to
   BNS 2023; **auto-apply only rows with `confidence==high`**, route the rest to a review list.
3. **ASR baseline.** Run faster-whisper (large-v3, int8) on ~20 FLEURS-ta clips and compute **WER/CER** (jiwer).

Wire steps 1–2 (text → statute-ID → BNS map) and the ASR step into a small LangGraph graph with one golden
`pytest` fixture. Print an F1 table and a WER number as the "it works" proof.

**Show me the plan first, then implement.** Keep everything local; update `PROGRESS.md` when you finish.
