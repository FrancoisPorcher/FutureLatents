#!/bin/bash
## Set up Slurm job
# Use the config name as the job name so %x expands correctly in filenames
#SBATCH --job-name=vjepa2_kinetics_400_deterministic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
# Save stdout and stderr in the submission directory before moving them
# to the experiment folder
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e

ROOT=/private/home/francoisporcher/FutureLatents
CONFIG_PATH=configs/vjepa2_kinetics_400_deterministic.yaml
CONFIG_NAME=$(basename "$CONFIG_PATH" .yaml)
EXPERIMENT_DIR="$ROOT/experiment/$CONFIG_NAME"
SLURM_LOG_DIR="$EXPERIMENT_DIR/slurm"

# Prepare experiment directory for SLURM logs
mkdir -p "$SLURM_LOG_DIR"
if [ -n "$SLURM_JOB_ID" ]; then
  OUT_FILE="$ROOT/${CONFIG_NAME}_${SLURM_JOB_ID}.out"
  ERR_FILE="$ROOT/${CONFIG_NAME}_${SLURM_JOB_ID}.err"
  [ -f "$OUT_FILE" ] && mv "$OUT_FILE" "$SLURM_LOG_DIR/"
  [ -f "$ERR_FILE" ] && mv "$ERR_FILE" "$SLURM_LOG_DIR/"
fi

cd "$ROOT"

# Silence bind warnings from non-interactive shells
source ~/.bashrc >/dev/null 2>&1
conda activate future_latents

echo "Conda environment: $CONDA_DEFAULT_ENV"
python --version
nvidia-smi

srun accelerate launch --num_processes 8 --num_machines 1 -m src.main \
  --config_path "$CONFIG_PATH"
