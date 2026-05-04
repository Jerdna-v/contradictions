#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --output=logs/models-%J.out
#SBATCH --error=logs/models-%J.err
#SBATCH --job-name="contradiction-models"

mkdir -p logs

SIF=/d/hpc/home/an49507/project/ul-fri-nlp-course-project-2025-2026-pb-j_enthusiast/containers/nlp-gpu.sif

singularity exec --nv --bind $SCRATCH:$SCRATCH $SIF \
    python3 -m vllm.entrypoints.openai.api_server \
        --model microsoft/Phi-3-mini-4k-instruct \
        --port 8001 --gpu-memory-utilization 0.3 &

singularity exec --nv --bind $SCRATCH:$SCRATCH $SIF \
    python3 -m vllm.entrypoints.openai.api_server \
        --model bigscience/bloomz-3b \
        --port 8002 --gpu-memory-utilization 0.25 &

singularity exec --nv --bind $SCRATCH:$SCRATCH $SIF \
    python3 -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --port 8003 --gpu-memory-utilization 0.35 &

singularity exec --nv --bind $SCRATCH:$SCRATCH $SIF \
    python3 -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-7B-Instruct \
        --port 8004 --gpu-memory-utilization 0.3 &

echo "Waiting for model servers to be ready..."
sleep 60
echo "Model servers running. Job will stay alive for wall time."
wait
