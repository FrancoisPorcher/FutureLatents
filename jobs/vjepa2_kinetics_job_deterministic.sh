#!/bin/bash
#SBATCH --job-name=vjepa2_kinetics
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

set -e

ROOT=/private/home/francoisporcher/FutureLatents
cd "$ROOT"

source ~/.bashrc
conda activate future_latents

echo "Conda environment: $CONDA_DEFAULT_ENV"
python --version
nvidia-smi

srun accelerate launch --num_processes 8 --num_machines 1 -m src.main \
  --config_path configs/vjepa2_kinetics_400_deterministic.yaml
