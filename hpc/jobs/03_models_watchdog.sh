#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --partition=all
#SBATCH --time=04:00:00
#SBATCH --output=logs/models-watchdog-%J.out
#SBATCH --error=logs/models-watchdog-%J.err
#SBATCH --job-name="contradiction-models-watchdog"

set -euo pipefail
mkdir -p logs

if [[ $# -lt 1 ]]; then
  echo "No model job IDs provided."
  exit 1
fi

job_ids=("$@")

is_terminal_failure() {
  case "$1" in
    FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

all_completed() {
  local state
  for jid in "${job_ids[@]}"; do
    state=$(sacct -j "$jid" --format=State --noheader | awk 'NR==1{print $1}')
    if [[ "$state" != "COMPLETED" ]]; then
      return 1
    fi
  done
  return 0
}

while true; do
  for jid in "${job_ids[@]}"; do
    state=$(sacct -j "$jid" --format=State --noheader | awk 'NR==1{print $1}')
    if [[ -z "$state" || "$state" == "PENDING" || "$state" == "RUNNING" ]]; then
      continue
    fi
    if is_terminal_failure "$state"; then
      echo "Detected failure in job $jid ($state). Cancelling sibling model jobs."
      for other in "${job_ids[@]}"; do
        if [[ "$other" != "$jid" ]]; then
          scancel "$other" || true
        fi
      done
      exit 1
    fi
  done

  if all_completed; then
    echo "All model jobs completed successfully."
    exit 0
  fi

  sleep 300
 done
