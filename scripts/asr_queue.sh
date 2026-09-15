#!/usr/bin/env bash
# ASR re-baseline after finding 6f (temperature fallback off; batched VAD-chunk
# decoding). Sequential GPU queue, one job at a time; logs to artifacts/logs/.
# Waits for a running profile_asr job to finish first.
#
#   Start-Process wsl.exe -ArgumentList '-d','Ubuntu','-e','bash','-lc',
#     '"cd /mnt/c/Users/madha/Downloads/fir-tn-capstone && bash scripts/asr_queue.sh"' -WindowStyle Hidden
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source "$HOME/.venvs/fir-tn/bin/activate"
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
mkdir -p artifacts/logs
Q=artifacts/logs/asr_queue.log
N="${N:-60}"

step() {  # step <name> <logfile> <command...>
  local name="$1" log="$2"; shift 2
  echo "$(date +%T) START  $name" >> "$Q"
  if "$@" > "$log" 2>&1; then
    echo "$(date +%T) OK     $name" >> "$Q"
  else
    echo "$(date +%T) FAILED $name (exit $?) -- see $log" >> "$Q"
  fi
}

while pgrep -f harness.profile_asr > /dev/null; do sleep 20; done

step "asr_baseline batched bs=8 temp0 (config default)" artifacts/logs/asr_baseline_batched.log \
  python -m harness.run_asr_baseline --n "$N"
step "asr_baseline sequential temp0" artifacts/logs/asr_baseline_seq.log \
  python -m harness.run_asr_baseline --n "$N" --batch-size 0 --tag seq
echo "$(date +%T) DONE" >> "$Q"
