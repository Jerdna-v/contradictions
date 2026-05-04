#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=all
#SBATCH --time=00:30:00
#SBATCH --output=logs/stage1-%J.out
#SBATCH --error=logs/stage1-%J.err
#SBATCH --job-name="contradiction-stage1"

mkdir -p logs

set -a && source .env && set +a

./.venv/bin/python scripts/run_pipeline.py \
    --env .env \
    --stages 1
