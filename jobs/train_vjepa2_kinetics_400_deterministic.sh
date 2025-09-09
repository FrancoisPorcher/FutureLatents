#!/bin/bash -l
#SBATCH --job-name=train_vjepa2_kinetics_400_deterministic
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=48:00:00
#SBATCH --partition=learnfair
#SBATCH --output=/private/home/francoisporcher/FutureLatents/experiment/%x/slurm/%x_%j.out
#SBATCH --error=/private/home/francoisporcher/FutureLatents/experiment/%x/slurm/%x_%j.err

ROOT=/private/home/francoisporcher/FutureLatents
CONFIG_PATH=configs/vjepa2_kinetics_400_deterministic.yaml
JOB_NAME=train_vjepa2_kinetics_400_deterministic
EXPERIMENT_DIR="$ROOT/experiment/$JOB_NAME"
SLURM_LOG_DIR="$EXPERIMENT_DIR/slurm"

echo "ROOT: $ROOT"
echo "CONFIG_PATH: $CONFIG_PATH"
echo "JOB_NAME: $JOB_NAME"
echo "EXPERIMENT_DIR: $EXPERIMENT_DIR"
echo "SLURM_LOG_DIR: $SLURM_LOG_DIR"

# Make experiment + log dirs
mkdir -p "$SLURM_LOG_DIR"

cd "$ROOT"
echo "Current directory: $(pwd)"

source /etc/profile.d/modules.sh

module load anaconda3/2023.03-1
module load cuda/11.8

eval "$(conda shell.bash hook)"
conda activate future_latents
echo "Conda environment: ${CONDA_DEFAULT_ENV:-unknown}"
python --version
nvidia-smi || true  # don't fail job if nvidia-smi is restricted

# Use Slurm's task count to drive accelerate for multi-node runs
# Redirect stderr to stdout so evaluation logs go to the .out file
accelerate launch --num_processes 8 --num_machines 2 -m src.main \
--config_path "$CONFIG_PATH" 2>&1

