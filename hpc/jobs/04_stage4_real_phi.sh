#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=06:00:00
#SBATCH --output=logs/stage4-real-phi-%J.out
#SBATCH --error=logs/stage4-real-phi-%J.err
#SBATCH --job-name="contradiction-stage4-real-phi"

set -euo pipefail

mkdir -p logs

cd /d/hpc/home/an49507/project/contradictions

SIF=/d/hpc/home/an49507/project/contradictions/containers/contradiction_pipeline.sif
VLLM_PORT=8001
SCRATCH_DIR=${SCRATCH:-/tmp}
CACHE_DIR=${SCRATCH_DIR}/hf

# Force stage4 to use the local self-hosted Phi-3 endpoint in this job.
export PHI3_ENDPOINT="http://127.0.0.1:${VLLM_PORT}/v1/completions"
export PHI3_MODEL_NAME="microsoft/Phi-3-mini-4k-instruct"
export USE_CELERY=false
export SINGULARITYENV_PYTHONNOUSERSITE=1
export SINGULARITYENV_HF_HOME="${CACHE_DIR}"
export SINGULARITYENV_TRANSFORMERS_CACHE="${CACHE_DIR}/transformers"
export SINGULARITYENV_HF_HUB_CACHE="${CACHE_DIR}/hub_cache"

echo "Starting local vLLM server on ${PHI3_ENDPOINT}"
singularity exec --cleanenv --nv --bind "${SCRATCH_DIR}:${SCRATCH_DIR}" "$SIF" \
  python3 -m vllm.entrypoints.openai.api_server \
    --model microsoft/Phi-3-mini-4k-instruct \
    --tokenizer-mode slow \
    --dtype half \
    --port "${VLLM_PORT}" \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    > "logs/vllm-phi-${SLURM_JOB_ID}.out" 2> "logs/vllm-phi-${SLURM_JOB_ID}.err" &

VLLM_PID=$!
trap 'kill ${VLLM_PID} >/dev/null 2>&1 || true' EXIT

echo "Waiting for vLLM readiness..."
for i in $(seq 1 120); do
  if python3 - <<'PY'
import json
import urllib.request
import urllib.error

url = "http://127.0.0.1:8001/v1/completions"
payload = {
    "model": "microsoft/Phi-3-mini-4k-instruct",
    "prompt": "ping",
    "max_tokens": 1,
    "temperature": 0.0,
}
req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
try:
    with urllib.request.urlopen(req, timeout=5) as r:
        ok = r.status in (200, 400)
except Exception:
    ok = False
raise SystemExit(0 if ok else 1)
PY
  then
    echo "vLLM is ready"
    break
  fi
  sleep 5
done

echo "Running stage 4 only with real self-hosted Phi-3"
./.venv/bin/python scripts/run_pipeline.py \
  --env /d/hpc/home/an49507/project/contradictions/.env \
  --stages 4

echo "Stage 4 run completed"