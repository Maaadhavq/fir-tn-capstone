#!/usr/bin/env bash
# Makefile targets without GNU make. This WSL has no `make` (installing it needs
# sudo), so `scripts/run.sh <target>` runs the same commands the Makefile would.
# If make is present it is simply used.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="$HOME/.venvs/fir-tn/bin/python"

if command -v make >/dev/null 2>&1; then
    exec make "$@"
fi

target="${1:-help}"; shift || true
case "$target" in
    help)         grep -E '^\s*@echo "' Makefile | sed -E 's/^\s*@echo "(.*)"$/\1/' ;;
    setup)        bash environment/setup_wsl.sh ;;
    data)         bash scripts/download_ilsi.sh && "$PY" scripts/download_fleurs.py --split test ;;
    schedule)     "$PY" scripts/build_bnss_schedule.py --check "$@" ;;
    bns-text)     "$PY" scripts/build_bns_text.py "$@" ;;
    slice)        "$PY" -m harness.run_slice "$@" ;;
    slice-full)   "$PY" -m harness.run_slice --full "$@" ;;
    statute)      "$PY" -m harness.run_statute_baseline "$@" ;;
    asr)          "$PY" -m harness.run_asr_baseline "$@" ;;
    asr-compare)  "$PY" -m harness.compare_asr_compute_types --n 10 "$@" ;;
    report)       "$PY" -m harness.run_slice --report-only ;;
    test)         "$PY" -m pytest tests/ -v "$@" ;;
    lint)         "$PY" -m ruff check src harness tests --select E,F,W,I --ignore E501 "$@" ;;
    extract-eval) "$PY" -m harness.run_extraction_eval "$@" ;;
    results)      "$PY" -m harness.summarize ;;
    serve)        cd src && exec "$PY" -m uvicorn fir.serving.app:app --host 127.0.0.1 --port 8000 ;;   # FIR_PREWARM=1 loads Whisper at startup
    clean)        rm -rf artifacts/models/* artifacts/reports/*; find . -name __pycache__ -type d -prune -exec rm -rf {} + ;;
    *)            echo "unknown target: $target" >&2; exit 2 ;;
esac
