#!/bin/bash

# Job configuration
#SBATCH --job-name=ddp-cifar10
#SBATCH --output=logs/ddp_%j.out
#SBATCH --error=logs/ddp_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=GPU-big
#SBATCH --gpus=4
#SBATCH --mem=64G

# Load modules and activate environment
source ~/.bashrc
conda activate py-ddp

# Get the directory where this script is located
SCRIPT_DIR="/lustre/scratch/cbm107c-ai_llm/gpu_performance_suite/coding/pytorch-tutorial"
cd "$SCRIPT_DIR"

# Create output directories
mkdir -p "$SCRIPT_DIR/logs"
mkdir -p "$SCRIPT_DIR/checkpoints"

# Verify GPU visibility
echo "=== System Information ==="
echo "Running on: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-Not set}"
echo "Working directory: $(pwd)"

# Print GPU information
echo -e "\n=== GPU Information ==="
nvidia-smi

# Start training
echo -e "\n=== Starting Training ==="
python train.py \
    --batch_size 128 \
    --epochs 50 \
    --lr 0.001 \
    --world_size 4

echo "Training completed. Check logs in $SCRIPT_DIR/logs/"
echo "Checkpoints saved to $SCRIPT_DIR/checkpoints/"

exit 0
