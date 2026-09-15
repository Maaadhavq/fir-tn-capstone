#!/usr/bin/env bash
# Sequential GPU experiment queue. One job at a time -- this box has 7.7 GB of
# WSL RAM and has rebooted its VM under concurrent memory pressure (PROGRESS.md
# finding 6e). Launch detached from Windows:
#
#   Start-Process wsl.exe -ArgumentList '-d','Ubuntu','-e','bash','-lc',
#     '"cd /mnt/c/Users/madha/Downloads/fir-tn-capstone && bash scripts/gpu_queue.sh"' -WindowStyle Hidden
#
# Every step logs to artifacts/logs/ and appends a STATUS line to queue.log so
# progress can be tailed from the Windows side without opening a wsl session.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source "$HOME/.venvs/fir-tn/bin/activate"
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
mkdir -p artifacts/logs
Q=artifacts/logs/queue.log

step() {  # step <name> <logfile> <command...>
  local name="$1" log="$2"; shift 2
  echo "$(date +%T) START  $name" >> "$Q"
  if "$@" > "$log" 2>&1; then
    echo "$(date +%T) OK     $name" >> "$Q"
  else
    echo "$(date +%T) FAILED $name (exit $?) -- see $log" >> "$Q"
  fi
  grep -E 'test micro|WER|best on dev|gain over' "$log" | sed 's/^/           /' >> "$Q"
}

echo "$(date +%T) QUEUE  begin" >> "$Q"

# 1. clean data-effect measurement: full corpus, default config
step "bert-full"        artifacts/logs/full_bert.log \
  python -m harness.run_statute_baseline --full --skip-tfidf

# 2. does averaging the two full-corpus models help?
step "ensemble-full"    artifacts/logs/ensemble_full.log \
  python -m harness.ensemble_eval --full

# 3-4. ablations on the 8k slice, one variable each (fast, ~12 min apiece)
step "bert-8k-headtail" artifacts/logs/bert_headtail.log \
  python -m harness.run_statute_baseline --skip-tfidf --tag head_tail --truncation head_tail

step "bert-8k-headlr10" artifacts/logs/bert_headlr10.log \
  python -m harness.run_statute_baseline --skip-tfidf --tag headlr10 --head-lr-mult 10

# 5. refresh the aggregate
step "results"          artifacts/logs/summarize.log \
  python -m harness.summarize

echo "$(date +%T) QUEUE  done" >> "$Q"
