#!/usr/bin/env bash
# Review-3 ASR sweep: decoder brakes on the replay loop (finding 6g), then an
# optional Tamil-fine-tuned Whisper run if scripts/download_whisper_tamil.py has
# left artifacts/models/.download_done. One GPU job at a time; logs in artifacts/logs/.
#
#   Start-Process wsl.exe -ArgumentList '-d','Ubuntu','-e','bash','-lc',
#     '"cd /mnt/c/Users/madha/Downloads/fir-tn-capstone && bash scripts/asr_queue_r3.sh"' -WindowStyle Hidden
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source "$HOME/.venvs/fir-tn/bin/activate"
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
mkdir -p artifacts/logs
Q=artifacts/logs/asr_queue_r3.log
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

while pgrep -f 'harness\.(profile_asr|run_asr_baseline)' > /dev/null; do sleep 20; done

step "rp11"     artifacts/logs/asr_rp11.log     python -m harness.run_asr_baseline --n "$N" --repetition-penalty 1.1 --tag rp11
step "rp12"     artifacts/logs/asr_rp12.log     python -m harness.run_asr_baseline --n "$N" --repetition-penalty 1.2 --tag rp12
step "nr5"      artifacts/logs/asr_nr5.log      python -m harness.run_asr_baseline --n "$N" --no-repeat-ngram-size 5 --tag nr5
step "rp11_nr5" artifacts/logs/asr_rp11_nr5.log python -m harness.run_asr_baseline --n "$N" --repetition-penalty 1.1 --no-repeat-ngram-size 5 --tag rp11_nr5

if [ -f artifacts/models/.download_done ]; then
  step "convert whisper-tamil-medium" artifacts/logs/whisper_tamil_convert.log \
    python scripts/download_whisper_tamil.py --convert-only
  if [ -d artifacts/models/whisper-tamil-medium-ct2 ]; then
    step "ft_medium" artifacts/logs/asr_ft_medium.log \
      python -m harness.run_asr_baseline --n "$N" --model artifacts/models/whisper-tamil-medium-ct2 --tag ft_medium
  fi
fi
echo "$(date +%T) DONE" >> "$Q"
