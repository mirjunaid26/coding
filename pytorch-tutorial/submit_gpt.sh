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

# Create output directories
mkdir -p logs
mkdir -p checkpoints

# Verify GPU visibility
echo "=== CUDA Visible Devices ==="
nvidia-smi

# Run the training script
python train_gpt.py \
    --data_path /path/to/your/training_data.txt \
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

echo "=== Training Completed ===
