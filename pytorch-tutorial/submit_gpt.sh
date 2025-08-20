#!/bin/bash

# Job configuration
#SBATCH --job-name=minigpt
#SBATCH --output=logs/gpt_%j.out
#SBATCH --error=logs/gpt_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --partition=GPU-big
#SBATCH --gpus=4
#SBATCH --mem=64G

# Load modules and activate environment
source ~/.bashrc
conda activate py-ddp

# Get the directory where this script is located
SCRIPT_DIR="/lustre/scratch/cbm107c-ai_llm/gpu_performance_suite/coding/pytorch-tutorial"
cd "$SCRIPT_DIR"

# Check if data path is provided, otherwise use default sample data
DATA_PATH="${1:-$SCRIPT_DIR/data/shakespeare/train.txt}"

# Create output directories in the working directory
mkdir -p "$SCRIPT_DIR/logs"
mkdir -p "$SCRIPT_DIR/checkpoints"

# Verify GPU visibility
echo "=== System Information ==="
echo "Running on: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-Not set}"
echo "Working directory: $(pwd)"
echo "Using data from: $DATA_PATH"

# Check if data file exists
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Data file not found at $DATA_PATH"
    echo "Available files in data directory:"
    ls -lh "$SCRIPT_DIR/data/" 2>/dev/null || echo "No data directory found"
    echo -e "\nPlease run 'python download_shakespeare.py' to download the Shakespeare dataset"
    echo "or provide a valid data path as an argument to this script."
    exit 1
fi

echo -e "\n=== GPU Information ==="
nvidia-smi

echo -e "\n=== Starting Training ==="

# Run the training script
python train_gpt.py \
    --data_path "$DATA_PATH" \
    --log_dir "$SCRIPT_DIR/logs" \
    --save_dir "$SCRIPT_DIR/checkpoints" \
    --batch_size 8 \
    --block_size 1024 \
    --d_model 768 \
    --n_layer 12 \
    --n_head 12 \
    --learning_rate 6e-4 \
    --max_steps 100000 \
    --log_interval 10 \
    --save_interval 1000 \
    --world_size 4

echo "Training completed. Check logs in $SCRIPT_DIR/logs/"
echo "Checkpoints saved to $SCRIPT_DIR/checkpoints/"

exit 0
