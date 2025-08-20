#!/bin/bash

# Check if data path is provided, otherwise use default sample data
DATA_PATH="${1:-data/sample_data.txt}"

# Convert to absolute path
DATA_PATH="$(cd "$(dirname "$DATA_PATH")"; pwd)/$(basename "$DATA_PATH")"

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

# Create output directories
mkdir -p logs
mkdir -p checkpoints

# Verify GPU visibility
echo "=== System Information ==="
echo "Running on: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-Not set}"
echo "Using data from: $DATA_PATH"
echo "Working directory: $(pwd)"

# Check if data file exists
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Data file not found at $DATA_PATH"
    echo "Please run 'python create_sample_data.py' to generate sample data or provide a valid data path"
    exit 1
fi

echo -e "\n=== GPU Information ==="
nvidia-smi

echo -e "\n=== Starting Training ==="

# Run the training script
python train_gpt.py \
    --data_path "$DATA_PATH" \
    --vocab_size 50000 \
    --d_model 768 \
    --n_layer 12 \
    --n_head 12 \
    --block_size 1024 \
    --batch_size 8 \
    --learning_rate 6e-4 \
    --max_steps 100000 \
    --log_interval 10 \
    --save_interval 1000 \
    --log_dir logs \
    --save_dir checkpoints \
    --world_size 4

RET_CODE=$?

if [ $RET_CODE -eq 0 ]; then
    echo -e "\n=== Training Completed Successfully ==="
else
    echo -e "\n!!! Training Failed with exit code $RET_CODE !!!"
    echo "Check the error log at: logs/gpt_${SLURM_JOB_ID:-UNKNOWN}.err"
    exit $RET_CODE
fi
