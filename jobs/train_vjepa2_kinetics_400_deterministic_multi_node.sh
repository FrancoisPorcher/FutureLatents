#!/bin/bash -l
#SBATCH --job-name=vjepa2_kinetics_400_deterministic
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1           # crucial: one launcher per node
#SBATCH --gpus-per-node=8
#SBATCH --time=48:00:00
#SBATCH --partition=learnfair
#SBATCH --output=/private/home/francoisporcher/FutureLatents/experiment/%x/slurm/%x_%j.out
#SBATCH --error=/private/home/francoisporcher/FutureLatents/experiment/%x/slurm/%x_%j.err

set -e

# --- Your paths/config ---
ROOT=/private/home/francoisporcher/FutureLatents
CONFIG_PATH=configs/vjepa2_kinetics_400_deterministic.yaml
CONFIG_NAME=$(basename "$CONFIG_PATH" .yaml)
EXPERIMENT_DIR="$ROOT/experiment/$CONFIG_NAME"
SLURM_LOG_DIR="$EXPERIMENT_DIR/slurm"
mkdir -p "$SLURM_LOG_DIR"

cd "$ROOT"
echo "ROOT: $ROOT"
echo "CONFIG_PATH: $CONFIG_PATH"
echo "CONFIG_NAME: $CONFIG_NAME"
echo "EXPERIMENT_DIR: $EXPERIMENT_DIR"
echo "SLURM_LOG_DIR: $SLURM_LOG_DIR"

# --- Env init (same as you had) ---
source /etc/profile.d/modules.sh
module load anaconda3/2023.03-1
module load cuda/11.8
eval "$(conda shell.bash hook)"
conda activate future_latents
python --version
nvidia-smi || true

# --- (Recommended) NCCL safety knobs ---
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Optional if you ever need to debug comms:
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO

echo "START TIME: $(date)"

# --- Multi-node parameters for Accelerate ---
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-8}
NNODES=${SLURM_NNODES}
NUM_PROCESSES=$(( NNODES * GPUS_PER_NODE ))
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=6000

# Build the accelerate launcher (note the escaped \$SLURM_PROCID)
export LAUNCHER="accelerate launch \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  --machine_rank \$SLURM_PROCID \
  --num_processes $NUM_PROCESSES \
  --num_machines $NNODES"

# Your program entrypoint (module) + args
export PROGRAM="-m src.main --config_path $CONFIG_PATH"

# Final command that each node will run
export CMD="$LAUNCHER $PROGRAM"

# Log everything additionally into your experiment folder
LOG_PATH="$SLURM_LOG_DIR/train_${SLURM_JOB_ID}.log"
srun --jobid "$SLURM_JOB_ID" bash -lc "$CMD" 2>&1 | tee -a "$LOG_PATH"
