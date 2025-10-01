#!/usr/bin/env bash
#SBATCH --job-name=ldm-cxr
#SBATCH --partition=bigbatch
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# ----------------------------
# Environment & Setup
# ----------------------------
WORKDIR="${SLURM_SUBMIT_DIR}"

# Change to the working directory to resolve relative file paths
cd "$WORKDIR"
echo "Working directory set to: $(pwd)"

# Add the working directory to PYTHONPATH to resolve module imports
export PYTHONPATH="$WORKDIR${PYTHONPATH:+:$PYTHONPATH}"

# Activate your conda/mamba environment
source ~/.bashrc
mamba activate "${ENV_NAME:?Must set ENV_NAME}"
echo "Activated environment: ${ENV_NAME}"

# ----------------------------
# Build the Python command arguments
# ----------------------------
ARGS=(
  # The '-t' flag enables training mode in main.py
  -t
  # The '--base' flag points to the YAML config
  --base "$CONFIG_PATH"
  # Set GPU count from SLURM environment
  --gpus "${SLURM_GPUS_ON_NODE:-1}"

  # --- Your Custom Dataset Arguments ---
  --data_root "$DATA_ROOT"
  --task "$TASK"
  --class_filter "$CLASS_FILTER"
)

# --- Overfitting Logic (from your cxr_ae.slurm) ---
if [[ "${OVERFIT_ONE:-0}" == "1" ]]; then
  ARGS+=( --overfit_one )
elif (( ${OVERFIT_K:-0} > 0 )); then
  ARGS+=( --overfit_k "$OVERFIT_K" )
fi
if [[ "${WANDB:-0}" == "1" ]]; then
  ARGS+=( --wandb )
  [[ -n "${WANDB_PROJECT:-}" ]] && ARGS+=( --wandb_project "$WANDB_PROJECT" )
  [[ -n "${WANDB_ENTITY:-}"  ]] && ARGS+=( --wandb_entity "$WANDB_ENTITY" )
  [[ -n "${WANDB_TAGS:-}"    ]] && ARGS+=( --wandb_tags "$WANDB_TAGS" )
fi
# ----------------------------
# Display Info & Run
# ----------------------------
echo "========================================="
echo "SLURM Job:  ${SLURM_JOB_ID:-<local>}"
echo "Config:     $CONFIG_PATH"
echo "Task:       $TASK (Class: $CLASS_FILTER)"
echo "Overfit:    (one=${OVERFIT_ONE:-0}, k=${OVERFIT_K:-0})"
echo "========================================="

# Launch the training job
python main.py "${ARGS[@]}"