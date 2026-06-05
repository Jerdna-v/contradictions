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

# Define explicit absolute paths based on your real project directory
BASE_DIR="/d/hpc/home/an49507/project/ul-fri-nlp-course-project-2025-2026-pb-j_enthusiast"
SIF="$BASE_DIR/containers/nlp-gpu.sif"

# Create a dedicated storage folder inside your project directory for cache and venv
STORE_DIR="$BASE_DIR/model_storage"
VLLM_VENV="$STORE_DIR/vllm_venv"
HF_HOME="$STORE_DIR/hf_home"
HF_HUB_CACHE="$HF_HOME/hub"
TRANSFORMERS_CACHE="$HF_HOME/transformers"

# Ensure all target directories exist on the host file system before running Singularity
mkdir -p "$VLLM_VENV"
mkdir -p "$HF_HUB_CACHE"
mkdir -p "$TRANSFORMERS_CACHE"

# 1. PRE-BUILD THE VIRTUAL ENVIRONMENT
echo "Checking and preparing vLLM virtual environment..."
singularity exec --nv --bind "$BASE_DIR":"$BASE_DIR" "$SIF" bash -c "
if [[ ! -x \"$VLLM_VENV/bin/python\" ]]; then \
    echo \"Creating new venv at $VLLM_VENV...\"; \
    python3 -m venv \"$VLLM_VENV\"; \
    \"$VLLM_VENV/bin/pip\" install --upgrade pip; \
    \"$VLLM_VENV/bin/pip\" install vllm; \
else \
    echo \"Virtual environment already exists at $VLLM_VENV.\"; \
fi \
"

# 2. SPIN UP THE MODEL SERVERS IN THE BACKGROUND
echo "Starting vLLM model servers..."

# Server 1: Phi-3
singularity exec --nv --bind "$BASE_DIR":"$BASE_DIR" "$SIF" bash -c "
export HF_HOME=\"$HF_HOME\"; \
export HF_HUB_CACHE=\"$HF_HUB_CACHE\"; \
export TRANSFORMERS_CACHE=\"$TRANSFORMERS_CACHE\"; \
exec \"$VLLM_VENV/bin/python\" -m vllm.entrypoints.openai.api_server \
    --model microsoft/Phi-3-mini-4k-instruct \
    --port 8001 --gpu-memory-utilization 0.3 --dtype=half \
" &

# Server 2: Bloomz
singularity exec --nv --bind "$BASE_DIR":"$BASE_DIR" "$SIF" bash -c "
export HF_HOME=\"$HF_HOME\"; \
export HF_HUB_CACHE=\"$HF_HUB_CACHE\"; \
export TRANSFORMERS_CACHE=\"$TRANSFORMERS_CACHE\"; \
exec \"$VLLM_VENV/bin/python\" -m vllm.entrypoints.openai.api_server \
    --model bigscience/bloomz-3b \
    --port 8002 --gpu-memory-utilization 0.25 --dtype=half \
" &

# Server 3: Llama-3.1
singularity exec --nv --bind "$BASE_DIR":"$BASE_DIR" "$SIF" bash -c "
export HF_HOME=\"$HF_HOME\"; \
export HF_HUB_CACHE=\"$HF_HUB_CACHE\"; \
export TRANSFORMERS_CACHE=\"$TRANSFORMERS_CACHE\"; \
exec \"$VLLM_VENV/bin/python\" -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --port 8003 --gpu-memory-utilization 0.35 --dtype=half \
" &

# Server 4: Qwen2.5
singularity exec --nv --bind "$BASE_DIR":"$BASE_DIR" "$SIF" bash -c "
export HF_HOME=\"$HF_HOME\"; \
export HF_HUB_CACHE=\"$HF_HUB_CACHE\"; \
export TRANSFORMERS_CACHE=\"$TRANSFORMERS_CACHE\"; \
exec \"$VLLM_VENV/bin/python\" -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8004 --gpu-memory-utilization 0.3 --dtype=half \
" &

echo "Waiting for model servers to initialize..."
sleep 60
echo "Model servers running. Job active."
wait
