# FIR-TN capstone. All targets run inside the WSL2 venv (environment/setup_wsl.sh).
PY := $(HOME)/.venvs/fir-tn/bin/python

.PHONY: help setup data schedule bns-text slice slice-full test lint asr asr-compare statute extract-eval report results serve clean

help:
	@echo "setup       install the Python 3.11 env (WSL2, uv)"
	@echo "data        download ILSI + FLEURS-ta"
	@echo "schedule    rebuild data/statutes/bnss_schedule1.csv from the gazette PDF (--check vs stub)"
	@echo "bns-text    rebuild data/statutes/bns_2023.jsonl (358 BNS sections) from the gazette PDF"
	@echo "slice       vertical slice: F1 table + WER (subsampled, minutes)"
	@echo "slice-full  same, on the entire ILSI corpus (hours)"
	@echo "statute     statute-ID baselines only"
	@echo "asr         ASR baseline only"
	@echo "report      re-print the slice summary without retraining"
	@echo "asr-compare sweep faster-whisper compute types (int8/float16/vad)"
	@echo "extract-eval per-slot P/R of the rule extractor on the gold smoke set"
	@echo "results     aggregate all JSON reports into artifacts/reports/RESULTS.md"
	@echo "serve       FastAPI on 127.0.0.1:8000 (local only)"
	@echo "test        golden regression suite"
	@echo "lint        ruff (E,F,W,I; line length ignored)"

setup:
	bash environment/setup_wsl.sh

data:
	bash scripts/download_ilsi.sh
	$(PY) scripts/download_fleurs.py --split test

schedule:
	$(PY) scripts/build_bnss_schedule.py --check

bns-text:
	$(PY) scripts/build_bns_text.py

slice:
	$(PY) -m harness.run_slice

slice-full:
	$(PY) -m harness.run_slice --full

statute:
	$(PY) -m harness.run_statute_baseline

asr:
	$(PY) -m harness.run_asr_baseline

asr-compare:
	$(PY) -m harness.compare_asr_compute_types --n 10

report:
	$(PY) -m harness.run_slice --report-only

test:
	$(PY) -m pytest tests/ -v

lint:
	$(PY) -m ruff check src harness tests --select E,F,W,I --ignore E501

extract-eval:
	$(PY) -m harness.run_extraction_eval

results:
	$(PY) -m harness.summarize

serve:
	cd src && $(PY) -m uvicorn fir.serving.app:app --host 127.0.0.1 --port 8000

clean:
	rm -rf artifacts/models/* artifacts/reports/*
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
