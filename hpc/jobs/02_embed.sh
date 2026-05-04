#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=all
#SBATCH --time=03:00:00
#SBATCH --output=logs/embed-%J.out
#SBATCH --error=logs/embed-%J.err
#SBATCH --job-name="contradiction-embed"

set -euo pipefail
mkdir -p logs

export PARALLEL_WORKERS=2
export USE_CELERY=false
export SQLITE_TIMEOUT_SEC=120
export SQLITE_BUSY_TIMEOUT_MS=120000
export SQLITE_WRITE_RETRIES=12
export SQLITE_RETRY_BASE_SEC=0.05
./.venv/bin/python scripts/run_pipeline.py --env .env --stages 2,3
