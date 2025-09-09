#!/bin/bash
#SBATCH --job-name=vjepa2_kinetics_400_deterministic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=48:00:00
# Save stdout and stderr in the submission directory before moving them
# to the experiment folder
#SBATCH --partition learnfair
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e

ROOT=/private/home/francoisporcher/FutureLatents
CONFIG_PATH=configs/vjepa2_kinetics_400_deterministic.yaml
JOB_NAME=$(basename "$0" .sh)
EXPERIMENT_DIR="$ROOT/experiment/$JOB_NAME"
SLURM_LOG_DIR="$EXPERIMENT_DIR/slurm"

echo "ROOT: $ROOT"
echo "CONFIG_PATH: $CONFIG_PATH"
echo "JOB_NAME: $JOB_NAME"
echo "EXPERIMENT_DIR: $EXPERIMENT_DIR"
echo "SLURM_LOG_DIR: $SLURM_LOG_DIR"

# Prepare experiment directory for SLURM logs
mkdir -p "$SLURM_LOG_DIR"

move_logs() {
  OUT_FILE="$SLURM_SUBMIT_DIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
  ERR_FILE="$SLURM_SUBMIT_DIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
  [ -f "$OUT_FILE" ] && mv "$OUT_FILE" "$SLURM_LOG_DIR/"
  [ -f "$ERR_FILE" ] && mv "$ERR_FILE" "$SLURM_LOG_DIR/"
}
trap move_logs EXIT

cd "$ROOT"
echo "Current directory: $(pwd)"

# Silence bind warnings from non-interactive shells
source ~/.bashrc >/dev/null 2>&1
conda activate future_latents

echo "Conda environment: $CONDA_DEFAULT_ENV"
python --version
nvidia-smi

srun accelerate launch --num_processes 8 --num_machines 1 -m src.main \
  --config_path "$CONFIG_PATH"
