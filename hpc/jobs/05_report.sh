#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=all
#SBATCH --time=00:30:00
#SBATCH --output=logs/report-%J.out
#SBATCH --error=logs/report-%J.err
#SBATCH --job-name="contradiction-report"

mkdir -p logs

set -a && source .env && set +a

./.venv/bin/python scripts/generate_report.py \
    --env .env \
    --output ./reports/contradiction_report.html

echo "Report written to $SCRATCH/contradiction_pipeline/report.html"
echo "Copy to local machine with:"
echo "  scp $USER@$(hostname -f):$SCRATCH/contradiction_pipeline/report.html ./report.html"
