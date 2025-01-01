#!/bin/bash
# sweep_agents.sh - Run sweep agents across GPUs

#SBATCH --job-name=sweep_agent
#SBATCH --output=logs/agent_%A_%a.out
#SBATCH --error=logs/agent_%A_%a.err
#SBATCH --time=14-00:00:00
#SBATCH --partition=gpu_batch
#SBATCH --nodes=1
#SBATCH --nodelist=GPUCBB8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --array=0-7%4

ulimit -n 65536

# Create necessary directories
mkdir -p logs

# Get the sweep ID
SWEEP_ID=$(cat current_sweep_id.txt)

# Activate conda and set up environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate workbook

# Set up wandb
export WANDB_API_KEY=ae44aed90a06b3d312be80d0dcdb489fcbc5b2c1
export WANDB_PROJECT=molmir_sweep
export WANDB_ENTITY=hani-goodarzi

# Set CUDA_VISIBLE_DEVICES to array task ID to distribute across GPUs
export CUDA_VISIBLE_DEVICES=${SLURM_ARRAY_TASK_ID}

echo "Starting agent ${SLURM_ARRAY_TASK_ID} for sweep ${SWEEP_ID}"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Running on node: $(hostname)"
echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"

# Run the agent
python run_sweep_agent.py \
    --config configs/sweep.yaml \
    --sweep_id "${SWEEP_ID}" \
    --agent_id "${SLURM_ARRAY_TASK_ID}"
