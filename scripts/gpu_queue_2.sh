#!/usr/bin/env bash
# Second GPU queue: the longer-training encoder run that the first queue's
# results point at. Same one-job-at-a-time discipline (PROGRESS.md 6e).
#
#   Start-Process wsl.exe -ArgumentList '-d','Ubuntu','-e','bash','-lc',
#     '"cd /mnt/c/Users/madha/Downloads/fir-tn-capstone && EPOCHS=4 EXTRA=\"--truncation head_tail --head-lr-mult 10\" bash scripts/gpu_queue_2.sh"' -WindowStyle Hidden
#
# Env:
#   EPOCHS  number of epochs for the full-corpus encoder (default 4)
#   EXTRA   extra flags for run_statute_baseline (ablation options that won on the 8k slice)
#   TAG     report/artifact tag (default: e<EPOCHS>)
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source "$HOME/.venvs/fir-tn/bin/activate"
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
mkdir -p artifacts/logs
Q=artifacts/logs/queue.log
EPOCHS="${EPOCHS:-4}"
EXTRA="${EXTRA:-}"
TAG="${TAG:-e${EPOCHS}}"

step() {
  local name="$1" log="$2"; shift 2
  echo "$(date +%T) START  $name" >> "$Q"
  if "$@" > "$log" 2>&1; then
    echo "$(date +%T) OK     $name" >> "$Q"
  else
    echo "$(date +%T) FAILED $name (exit $?) -- see $log" >> "$Q"
  fi
  grep -E 'dev micro|test micro|trained in|best on dev|gain over' "$log" | sed 's/^/           /' >> "$Q"
}

echo "$(date +%T) QUEUE2 begin (epochs=$EPOCHS extra='$EXTRA' tag=$TAG)" >> "$Q"

# shellcheck disable=SC2086
step "bert-full-$TAG" "artifacts/logs/full_bert_$TAG.log" \
  python -m harness.run_statute_baseline --full --skip-tfidf --epochs "$EPOCHS" --tag "$TAG" $EXTRA

step "ensemble-full-$TAG" "artifacts/logs/ensemble_full_$TAG.log" \
  python -m harness.ensemble_eval --full --bert-tag "$TAG"

step "results" artifacts/logs/summarize.log python -m harness.summarize

echo "$(date +%T) QUEUE2 done" >> "$Q"
