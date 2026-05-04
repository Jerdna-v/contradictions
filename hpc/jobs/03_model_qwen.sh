#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --output=logs/models-qwen-%J.out
#SBATCH --error=logs/models-qwen-%J.err
#SBATCH --job-name="contradiction-model-qwen"

mkdir -p logs

SIF=/d/hpc/home/an49507/project/ul-fri-nlp-course-project-2025-2026-pb-j_enthusiast/containers/nlp-gpu.sif

singularity exec --nv --bind $SCRATCH:$SCRATCH $SIF \
    python3 -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-7B-Instruct \
        --port 8004 --gpu-memory-utilization 0.3
