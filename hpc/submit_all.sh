#!/bin/bash
set -e

JOB1=$(sbatch --parsable hpc/jobs/01_stage1.sh)
echo "Submitted stage1 job: $JOB1"

JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 hpc/jobs/02_embed.sh)
echo "Submitted embed job: $JOB2"

JOB3_PHI=$(sbatch --parsable hpc/jobs/03_model_phi.sh)
echo "Submitted model phi job: $JOB3_PHI"
JOB3_BLOOMZ=$(sbatch --parsable hpc/jobs/03_model_bloomz.sh)
echo "Submitted model bloomz job: $JOB3_BLOOMZ"
JOB3_LLAMA=$(sbatch --parsable hpc/jobs/03_model_llama.sh)
echo "Submitted model llama job: $JOB3_LLAMA"
JOB3_QWEN=$(sbatch --parsable hpc/jobs/03_model_qwen.sh)
echo "Submitted model qwen job: $JOB3_QWEN"

JOB4=$(sbatch --parsable --dependency=afterok:$JOB2:$JOB3_PHI:$JOB3_BLOOMZ:$JOB3_LLAMA:$JOB3_QWEN hpc/jobs/04_pipeline.sh)
echo "Submitted pipeline job: $JOB4"

JOB5=$(sbatch --parsable --dependency=afterok:$JOB4 hpc/jobs/05_report.sh)
echo "Submitted report job: $JOB5"

echo "All jobs submitted. Monitor with: squeue -u $USER"
