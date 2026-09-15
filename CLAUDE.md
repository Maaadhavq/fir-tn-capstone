# CLAUDE.md — FIR TN Capstone (Claude Code)

@CONTEXT.md

## Working notes for Claude Code
- Phase: **implementation started (2026-08-20).** The repo skeleton, the WSL2 Python 3.11 environment,
  and the first vertical slice (IMPLEMENTATION.md §4) are built. Planning docs remain authoritative for
  everything not yet built.
- Canonical deliverables: PLAN.md (roadmap), ARCHITECTURE.md, IMPLEMENTATION.md (engineering plan),
  DATASETS.md (verified data), OPEN-QUESTIONS.md. CONTEXT.md is the single source of truth for project
  facts; this file only imports it.
- PROGRESS.md tracks live status — read it first in a new session.

## How to run anything here
All Python runs inside WSL2 Ubuntu, **not** native Windows — Smart App Control blocks `pyarrow`.
See `environment/README.md`. The venv is at `~/.venvs/fir-tn` (WSL ext4, not `/mnt/c`).

```bash
wsl -d Ubuntu -e bash -lc "cd /mnt/c/Users/madha/Downloads/fir-tn-capstone && source ~/.venvs/fir-tn/bin/activate && <command>"
```

`make test` (2s, no weights needed) · `make slice` (F1 table + WER) · `make slice-full`.

## Conventions that matter
- **Never add HF `datasets` or `bitsandbytes` to the Windows-side stack** — both pull in `pyarrow`.
  Loaders read plain JSONL/TSV/tar; use DuckDB for parquet.
- **The IPC→BNS auto-apply rule is a legal-safety invariant, not a tunable.** Only
  `confidence == high AND needs_review == N AND target is a real BNS section` may be applied without a
  human. `tests/test_slice_golden.py` pins this; if a change makes those tests fail, the change is
  wrong unless the legal position actually changed.
- **Cognizability never defaults.** An unknown section routes to the officer, never silently to CSR.
- Decision-support only: the officer is the author of record. Nothing is ever filed automatically.
