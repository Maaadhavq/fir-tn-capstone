# Environment

## Why WSL2

Smart App Control on this Windows host blocks `pyarrow`. HuggingFace `datasets`
and `bitsandbytes` both depend on it, so the data and training stack cannot be
installed natively. A WSL2 Ubuntu userland sidesteps the block completely and is
still 100% local — no hosted APIs are used anywhere in this project.

Two further precautions keep the pyarrow surface at zero even inside WSL, so the
code stays runnable natively if anyone needs it to be:

- every loader reads plain JSONL / TSV / tar, never parquet (`duckdb` is pinned
  for any parquet we hit later);
- the InLegalBERT trainer is a plain torch loop, not HF `Trainer`.

## Setup

```bash
bash environment/setup_wsl.sh
```

Installs `uv` into `~/.local/bin` (no sudo needed — this box has no passwordless
sudo, so `apt install python3.11` is not available), fetches a standalone Python
3.11, and creates the venv at `~/.venvs/fir-tn`.

The venv lives on WSL's ext4 filesystem, **not** under `/mnt/c`. Python imports
across the 9p mount are roughly an order of magnitude slower.

```bash
source ~/.venvs/fir-tn/bin/activate
```

## Verified on this machine (2026-08-20)

| | |
|---|---|
| OS | Ubuntu 24.04.1 LTS under WSL2 |
| Python | 3.11.15 (uv standalone) |
| torch | 2.11.0+cu128 |
| GPU | RTX 5050 Laptop, **8.5 GB**, sm_120 (Blackwell) |
| Host RAM | 15.6 GB (WSL sees ~7.7 GB) |

### Two hardware notes that shaped the pins

**The GPU is 8 GB, not the 16 GB in CONTEXT.md's locked decisions.** The vertical
slice fits comfortably. What does *not* fit is Qwen2.5-14B 4-bit (~9 GB), the
planned synthetic-data generator — see `configs/models.yaml`, where every model
is annotated with `fits: true|false` against this budget, and PROGRESS.md, where
the decision is logged as open.

**Blackwell needs CUDA 12.8+.** `sm_120` has no kernels in cu121 wheels, which is
why `setup_wsl.sh` installs torch from the cu128 index explicitly rather than
letting pip resolve it. CTranslate2 (faster-whisper's backend) is compiled
per-architecture too, so `src/fir/asr/whisper.py` falls back to CPU int8 rather
than crashing if the installed wheel predates sm_120.

## Operating rules for long jobs on this box (each learned the hard way)

WSL gets ~7.7 GB of the host's 15.6 GB. The GPU has 8 GB. RAM, not VRAM, is the
binding constraint for anything that touches the full ILSI corpus.

1. **One memory-heavy job at a time.** Whisper (~3 GB RSS) alongside a full-corpus
   TF-IDF fit was OOM-killed. BERT training alongside anything else is not worth
   the risk either.
2. **Launch long jobs so they survive the launching session.** `nohup … &` inside
   `wsl -e bash -lc` does *not* — WSL kills the tree when the session returns.
   Either keep the `wsl` invocation attached for the whole run, or detach fully
   from the Windows side:
   ```powershell
   Start-Process wsl.exe -ArgumentList '-d','Ubuntu','-e','bash','-lc','"cd … && python -u -m harness.run_statute_baseline --full > artifacts/logs/run.log 2>&1"' -WindowStyle Hidden
   ```
3. **Log to `artifacts/logs/` on `/mnt/c`, never `/tmp`.** The VM restarted twice
   in one session and `/tmp` went with it.
4. **Monitor from the Windows side while a heavy job runs** — tail the log on
   `C:`, watch `vmmemWSL` in Task Manager. Avoid piling `wsl` calls onto a VM
   that is already near its memory ceiling.
5. **The VM has restarted itself under the 42k-doc vectorization** with no OOM
   record, no traceback and no crash dump (PROGRESS.md finding 6e). Chunked
   `HashingVectorizer` + a detached launch got through; the root cause is not
   established. If it recurs, the fix to try first is more VM memory:
   `%USERPROFILE%\.wslconfig` →
   ```ini
   [wsl2]
   memory=12GB
   swap=4GB
   ```
   then `wsl --shutdown`. Not applied yet — it is a persistent config change.

## Data

```bash
make data
```

| | Source | Size |
|---|---|---|
| ILSI train/dev/test | Zenodo 6053791 | 512 MB |
| FLEURS ta_in test | HF `google/fleurs` | 407 MB |
| InLegalBERT | HF `law-ai/InLegalBERT` | 534 MB |
| faster-whisper large-v3 | HF `Systran/faster-whisper-large-v3` | 3.09 GB |

Raw data is gitignored. `data/mapping/` is the exception — it is our own
hand-built artifact and is committed.

## Running

```bash
make test        # golden regression, no weights needed, ~2s
make lint        # ruff (E,F,W,I; line length ignored)
make slice       # F1 table + WER, subsampled
make slice-full  # entire ILSI corpus
```

This WSL image has **no `make`** (installing it needs `sudo apt install make`).
`scripts/run.sh <target>` runs the identical commands for every Makefile target
and simply execs `make` when it is present:

```bash
bash scripts/run.sh test -q
```

Line endings are LF everywhere (`.gitattributes`); editing from the Windows side
with tools that write CRLF breaks the Makefile and shell scripts in WSL.
