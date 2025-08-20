# MiniGPT: A Minimal PyTorch Implementation of GPT with DDP

This is a minimal implementation of the GPT (Generative Pre-trained Transformer) model, inspired by Andrej Karpathy's [minGPT](https://github.com/karpathy/minGPT), with added support for Distributed Data Parallel (DDP) training across multiple GPUs.

## Features

- Minimal and clean implementation of GPT architecture
- Multi-GPU training with PyTorch DDP
- Efficient data loading with proper batching
- Checkpointing and resuming training
- Text generation with temperature and top-k sampling
- TensorBoard integration for training monitoring

## Requirements

- Python 3.8+
- PyTorch 1.12+ with CUDA
- torchtext
- tqdm
- tensorboard

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/minigpt-ddp.git
cd minigpt-ddp
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training

### Prepare Your Data

Place your training data in a text file (e.g., `data/input.txt`). The model will automatically handle the tokenization.

### Start Training

To train the model on a single machine with 4 GPUs:

```bash
# Make the submission script executable
chmod +x submit_gpt.sh

# Submit the job
sbatch submit_gpt.sh
```

### Training Parameters

You can modify the training parameters in `submit_gpt.sh` or pass them directly to `train_gpt.py`:

- `--data_path`: Path to training data file
- `--vocab_size`: Vocabulary size (default: 50000)
- `--d_model`: Model dimension (default: 768)
- `--n_layer`: Number of transformer layers (default: 12)
- `--n_head`: Number of attention heads (default: 12)
- `--block_size`: Context length (default: 1024)
- `--batch_size`: Batch size per GPU (default: 8)
- `--learning_rate`: Learning rate (default: 6e-4)
- `--max_steps`: Maximum number of training steps (default: 100000)
- `--save_dir`: Directory to save checkpoints (default: 'checkpoints')

## Text Generation

To generate text using a trained model:

```bash
python generate.py \
    --checkpoint checkpoints/ckpt_step_100000.pt \
    --prompt "Once upon a time" \
    --max_tokens 1000 \
    --temperature 0.8 \
    --top_k 40
```

## Model Architecture

The implementation follows the original GPT architecture:

- Multi-head self-attention with causal masking
- Position-wise feed-forward networks
- Layer normalization and residual connections
- Learned position embeddings

## Distributed Training

The implementation uses PyTorch's DistributedDataParallel (DDP) for multi-GPU training. Key features:

- Data parallelism across multiple GPUs
- Gradient synchronization across processes
- Efficient memory usage with gradient checkpointing

## Monitoring Training

Training progress can be monitored using TensorBoard:

```bash
tensorboard --logdir=logs
```

Then open `http://localhost:6006` in your browser.

## Checkpoints

Checkpoints are saved in the `checkpoints` directory with the following format:
- `ckpt_step_XXXXXX.pt`: Model checkpoint at step XXXXXX

Each checkpoint contains:
- Model state dict
- Optimizer state
- Training configuration
- Current step number

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [minGPT](https://github.com/karpathy/minGPT)
- [nanoGPT](https://github.com/karpathy/nanoGPT)
