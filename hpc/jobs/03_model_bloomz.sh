#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --output=logs/models-bloomz-%J.out
#SBATCH --error=logs/models-bloomz-%J.err
#SBATCH --job-name="contradiction-model-bloomz"

set -euo pipefail
mkdir -p logs

SIF=/d/hpc/home/an49507/project/ul-fri-nlp-course-project-2025-2026-pb-j_enthusiast/containers/nlp-gpu.sif
VLLM_VENV="$SCRATCH/vllm_venv"

singularity exec --nv --bind $SCRATCH:$SCRATCH $SIF bash -lc "\
VENV=\"$VLLM_VENV\"; \
if [[ ! -x \"\$VENV/bin/python\" ]]; then \
    python3 -m venv \"\$VENV\"; \
    \"\$VENV/bin/pip\" install --upgrade pip; \
    \"\$VENV/bin/pip\" install vllm; \
fi; \
exec \"\$VENV/bin/python\" -m vllm.entrypoints.openai.api_server \
    --model bigscience/bloomz-3b \
    --port 8002 --gpu-memory-utilization 0.25\
"
