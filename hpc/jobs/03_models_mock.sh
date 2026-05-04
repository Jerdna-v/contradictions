#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --partition=all
#SBATCH --time=04:00:00
#SBATCH --output=logs/models-mock-%J.out
#SBATCH --error=logs/models-mock-%J.err
#SBATCH --job-name="contradiction-models-mock"

set -euo pipefail
mkdir -p logs

PY=$(which python3 || echo python3)
# Use working directory (SLURM sets WorkDir to repository path) to find scripts
SCRIPTS_DIR="$PWD/scripts"

PIDS=""
start_mock() {
  port=$1
  log_prefix="logs/mock-${port}"
  "$PY" "$SCRIPTS_DIR/mock_model_server.py" --port "$port" > "${log_prefix}.out" 2> "${log_prefix}.err" &
  PIDS="$PIDS $!"
}

start_mock 8001
start_mock 8002
start_mock 8003
start_mock 8004

trap 'echo "Stopping mocks..."; kill $PIDS || true; wait' SIGTERM SIGINT

echo "Started mock servers (pids:$PIDS)"
wait
