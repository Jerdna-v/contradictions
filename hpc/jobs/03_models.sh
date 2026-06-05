#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --output=logs/models-%J.out
#SBATCH --error=logs/models-%J.err
#SBATCH --job-name="contradiction-models"

mkdir -p logs

SIF=/d/hpc/home/an49507/project/ul-fri-nlp-course-project-2025-2026-pb-j_enthusiast/containers/nlp-gpu.sif
VLLM_VENV="$SCRATCH/vllm_venv"
HF_HOME="$SCRATCH/hf_home"
HF_HUB_CACHE="$HF_HOME/hub"
TRANSFORMERS_CACHE="$HF_HOME/transformers"

singularity exec --nv --bind $SCRATCH:$SCRATCH $SIF bash -lc "\
export HF_HOME=\"$HF_HOME\"; \
export HF_HUB_CACHE=\"$HF_HUB_CACHE\"; \
export TRANSFORMERS_CACHE=\"$TRANSFORMERS_CACHE\"; \
VENV=\"$VLLM_VENV\"; \
if [[ ! -x \"\$VENV/bin/python\" ]]; then \
    python3 -m venv \"\$VENV\"; \
    \"\$VENV/bin/pip\" install --upgrade pip; \
    \"\$VENV/bin/pip\" install vllm; \
fi; \
exec \"\$VENV/bin/python\" -m vllm.entrypoints.openai.api_server \
    --model microsoft/Phi-3-mini-4k-instruct \
    --port 8001 --gpu-memory-utilization 0.3 --dtype=half\
" &

singularity exec --nv --bind $SCRATCH:$SCRATCH $SIF bash -lc "\
export HF_HOME=\"$HF_HOME\"; \
export HF_HUB_CACHE=\"$HF_HUB_CACHE\"; \
export TRANSFORMERS_CACHE=\"$TRANSFORMERS_CACHE\"; \
VENV=\"$VLLM_VENV\"; \
if [[ ! -x \"\$VENV/bin/python\" ]]; then \
    python3 -m venv \"\$VENV\"; \
    \"\$VENV/bin/pip\" install --upgrade pip; \
    \"\$VENV/bin/pip\" install vllm; \
fi; \
exec \"\$VENV/bin/python\" -m vllm.entrypoints.openai.api_server \
    --model bigscience/bloomz-3b \
    --port 8002 --gpu-memory-utilization 0.25 --dtype=half\
" &

singularity exec --nv --bind $SCRATCH:$SCRATCH $SIF bash -lc "\
export HF_HOME=\"$HF_HOME\"; \
export HF_HUB_CACHE=\"$HF_HUB_CACHE\"; \
export TRANSFORMERS_CACHE=\"$TRANSFORMERS_CACHE\"; \
VENV=\"$VLLM_VENV\"; \
if [[ ! -x \"\$VENV/bin/python\" ]]; then \
    python3 -m venv \"\$VENV\"; \
    \"\$VENV/bin/pip\" install --upgrade pip; \
    \"\$VENV/bin/pip\" install vllm; \
fi; \
exec \"\$VENV/bin/python\" -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --port 8003 --gpu-memory-utilization 0.35 --dtype=half\
" &

singularity exec --nv --bind $SCRATCH:$SCRATCH $SIF bash -lc "\
export HF_HOME=\"$HF_HOME\"; \
export HF_HUB_CACHE=\"$HF_HUB_CACHE\"; \
export TRANSFORMERS_CACHE=\"$TRANSFORMERS_CACHE\"; \
VENV=\"$VLLM_VENV\"; \
if [[ ! -x \"\$VENV/bin/python\" ]]; then \
    python3 -m venv \"\$VENV\"; \
    \"\$VENV/bin/pip\" install --upgrade pip; \
    \"\$VENV/bin/pip\" install vllm; \
fi; \
exec \"\$VENV/bin/python\" -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8004 --gpu-memory-utilization 0.3 --dtype=half\
" &

echo "Waiting for model servers to be ready..."
sleep 60
echo "Model servers running. Job will stay alive for wall time."
wait
