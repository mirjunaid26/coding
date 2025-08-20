#!/bin/bash

# Job configuration
#SBATCH --job-name=nn_ddp
#SBATCH --output=results/nn_ddp_%j.out
#SBATCH --error=results/nn_ddp_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --partition=GPU-big
#SBATCH --gpus=4
#SBATCH --mem=16G

# Load modules and activate environment
source ~/.bashrc
conda activate py-ddp

# Verify GPU visibility
echo "=== CUDA Visible Devices ==="
nvidia-smi

# Set the number of processes per node
export GPUS_PER_NODE=4

echo "=== Starting DDP Training ==="
echo "Using $GPUS_PER_NODE GPUs per node"

# Run the training with torchrun
torchrun --nproc_per_node=$GPUS_PER_NODE \
         --nnodes=1 \
         --node_rank=0 \
         --master_addr=$(hostname) \
         --master_port=12345 \
         nn_ddp.py

echo "=== Training Completed ==="
