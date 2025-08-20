# MiniGPT: Distributed Training with PyTorch DDP

This project implements a minimal GPT (Generative Pre-trained Transformer) model with PyTorch's Distributed Data Parallel (DDP) for efficient multi-GPU training on HPC clusters.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Getting Started](#getting-started)
- [Training on HPC](#training-on-hpc)
- [Monitoring Training](#monitoring-training)
- [Generating Text](#generating-text)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## Overview

### GPT (Generative Pre-trained Transformer)
GPT is a transformer-based language model that uses self-attention mechanisms to process sequential data. This implementation includes:
- Multi-head self-attention
- Position-wise feed-forward networks
- Layer normalization and residual connections
- Learned position embeddings

### PyTorch DDP (Distributed Data Parallel)
PyTorch DDP enables efficient multi-GPU training by:
- Replicating the model across multiple GPUs
- Splitting the input data across GPUs
- Synchronizing gradients during backpropagation
- Aggregating model updates

## Project Structure

```
.
├── data.py            # Data loading and preprocessing
├── minigpt.py         # GPT model implementation
├── train_gpt.py       # Training script with DDP support
├── generate.py        # Text generation script
├── submit_gpt.sh      # SLURM submission script
├── requirements.txt   # Python dependencies
└── checkpoints/       # Directory for model checkpoints
└── logs/              # Training logs and TensorBoard files
└── README.md          # This file
```

## Dependencies

- Python 3.8+
- PyTorch 1.12+ with CUDA
- torchtext
- tqdm
- tensorboard

Install dependencies:
```bash
pip install -r requirements.txt
```

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/yourusername/minigpt-ddp.git
cd minigpt-ddp
```

2. Prepare your training data as a text file (e.g., `data/input.txt`)

## Training on HPC

### SLURM Submission

1. Make the submission script executable:
```bash
chmod +x submit_gpt.sh
```

2. Submit the job with your training data:
```bash
sbatch submit_gpt.sh /path/to/your/training_data.txt
```

### Customizing Training

Edit `submit_gpt.sh` to modify training parameters:
- `--batch_size`: Batch size per GPU
- `--d_model`: Model dimension
- `--n_layer`: Number of transformer layers
- `--n_head`: Number of attention heads
- `--block_size`: Context length
- `--learning_rate`: Learning rate
- `--max_steps`: Maximum training steps

## Monitoring Training

### Checking Job Status

```bash
squeue --me
```

### Viewing GPU Utilization

```bash
# On the compute node
nvidia-smi -l 2  # Updates every 2 seconds
```

### Viewing Logs

```bash
# View output log
tail -f logs/gpt_<jobid>.out

# View error log
tail -f logs/gpt_<jobid>.err
```

### TensorBoard

```bash
# On your local machine
ssh -L 6006:localhost:6006 <username>@login.hpc.example.com

# On the login node
cd /path/to/project
tensorboard --logdir=logs --port=6006
```
Then open `http://localhost:6006` in your browser.

## Generating Text

After training, generate text using a checkpoint:

```bash
python generate.py \
    --checkpoint checkpoints/ckpt_step_100000.pt \
    --prompt "Your prompt here" \
    --max_tokens 1000 \
    --temperature 0.8 \
    --top_k 40
```

## Troubleshooting

### Common Issues

1. **File Not Found**
   - Ensure the data path is accessible from all nodes
   - Use absolute paths in the submission script

2. **Out of Memory**
   - Reduce `batch_size` or `block_size`
   - Enable gradient checkpointing

3. **DDP Initialization Errors**
   - Make sure all processes can communicate
   - Check firewall settings

### Checking System Status

```bash
# Check disk space
df -h

# Check CPU and memory usage
top

# Check GPU status
nvidia-smi
```

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [PyTorch DDP Documentation](https://pytorch.org/docs/stable/notes/ddp.html)
- [minGPT](https://github.com/karpathy/minGPT)
- [nanoGPT](https://github.com/karpathy/nanoGPT)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
