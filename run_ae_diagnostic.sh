#!/usr/bin/env bash
#
# Launches a DIAGNOSTIC training job for the Autoencoder on a single TB sample.
#

set -euo pipefail

# REQUIRED: Set the name of your conda or mamba environment
export ENV_NAME="ldm" # Or whatever your environment is named

# --- Key Training Parameters ---
export CONFIG_PATH="configs/autoencoder/autoencoder_kl_8x8x64.yaml"
export DATA_ROOT="/datasets/mmolefe/cleaned"
export TASK="TB"
export CLASS_FILTER="1" # 1 for diseased, 0 for normal


# --- W&B Configuration ---
export WANDB=1 # Set to 1 to enable logging
export WANDB_PROJECT="cxr-autoencoder-diagnostics"
export WANDB_TAGS="autoencoder,diagnostic,overfit"

# --- Enable overfitting on one sample for fast feedback ---
export OVERFIT_ONE=1
export OVERFIT_K=0

# --- Submit the SLURM job ---
echo "Submitting SLURM job for Autoencoder DIAGNOSTIC training..."
echo "------------------------------------------------"
echo "  ▶️  Config:    $CONFIG_PATH"
echo "  ▶️  Task:      $TASK (Class: $CLASS_FILTER)"
echo "  ▶️  Overfit:   ENABLED (1 sample)"
echo "------------------------------------------------"
sbatch ldm_train.slurm
echo "✅ Job successfully submitted!"