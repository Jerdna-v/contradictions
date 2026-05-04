#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --output=logs/pipeline-%J.out
#SBATCH --error=logs/pipeline-%J.err
#SBATCH --job-name="contradiction-pipeline"

mkdir -p logs

set -a && source .env && set +a
export USE_CELERY=false

./.venv/bin/python scripts/run_pipeline.py \
    --env .env \
    --stages 4,5,6,7
