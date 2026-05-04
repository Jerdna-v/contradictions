#!/bin/bash
#SBATCH --job-name=test-stage4-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --output=/d/hpc/home/an49507/project/contradictions/logs/test-stage4-single-%j.out
#SBATCH --error=/d/hpc/home/an49507/project/contradictions/logs/test-stage4-single-%j.err
#SBATCH --partition=gpu

set -e

cd /d/hpc/home/an49507/project/contradictions

# Load environment variables
source /d/hpc/home/an49507/project/contradictions/.env

export SINGULARITYENV_PYTHONNOUSERSITE=1
export APPTAINERENV_HF_HOME=/tmp/hf_cache
mkdir -p /tmp/hf_cache
chmod 777 /tmp/hf_cache

SIF=/d/hpc/home/an49507/project/contradictions/containers/contradiction_pipeline.sif

echo "=== Test Stage4 Single Pair ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
echo "Container: $SIF"
echo "HF Cache: /tmp/hf_cache"
echo "Available space in /tmp:"
df -h /tmp | tail -1
echo "Starting vLLM server on GPU..."

# Launch vLLM server in background with explicit bind mounts
apptainer exec --nv \
  --bind /tmp:/tmp \
  --env HF_HOME=/tmp/hf_cache \
  $SIF python -m vllm.entrypoints.openai.api_server \
  --model microsoft/Phi-3-mini-4k-instruct \
  --tensor-parallel-size 1 \
  --tokenizer-mode slow \
  --dtype half \
  --port 8001 \
  > logs/vllm-test-${SLURM_JOB_ID}.out 2> logs/vllm-test-${SLURM_JOB_ID}.err &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# Wait for vLLM to be ready
echo "Waiting for vLLM server to initialize..."
for i in {1..60}; do
  if curl -s -X POST http://127.0.0.1:8001/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"microsoft/Phi-3-mini-4k-instruct","prompt":"test","max_tokens":1}' > /dev/null 2>&1; then
    echo "vLLM server is ready after $i seconds"
    break
  fi
  if [ $i -eq 60 ]; then
    echo "ERROR: vLLM server did not become ready after 60 seconds"
    kill $VLLM_PID || true
    exit 1
  fi
  sleep 1
done

echo ""
echo "=== Running test script ==="
apptainer exec --nv $SIF python /d/hpc/home/an49507/project/contradictions/scripts/test_stage4_single_pair.py

TEST_EXIT=$?

echo ""
echo "Test exit code: $TEST_EXIT"

# Cleanup
echo "Stopping vLLM server..."
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

exit $TEST_EXIT
