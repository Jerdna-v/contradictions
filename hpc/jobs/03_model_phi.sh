#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --output=logs/models-phi-%J.out
#SBATCH --error=logs/models-phi-%J.err
#SBATCH --job-name="contradiction-model-phi"

mkdir -p logs

SIF=/d/hpc/home/an49507/project/ul-fri-nlp-course-project-2025-2026-pb-j_enthusiast/containers/nlp-gpu.sif

singularity exec --nv --bind $SCRATCH:$SCRATCH $SIF \
    python3 -m vllm.entrypoints.openai.api_server \
        --model microsoft/Phi-3-mini-4k-instruct \
        --port 8001 --gpu-memory-utilization 0.3
