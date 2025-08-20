# PyTorch DDP Quickstart for amplitUDE HPC

This guide provides a quickstart for running distributed PyTorch training using DistributedDataParallel (DDP) on the amplitUDE HPC system.

## Prerequisites

- Access to amplitUDE HPC cluster
- Python 3.8+
- PyTorch 2.0+ with CUDA support
- SLURM workload manager (for job submission)

## Installation

1. Load required modules:
```bash
module load python/3.10
module load cuda/11.8
module load pytorch/2.0.1
```

2. Create and activate a virtual environment:
```bash
python -m venv ddp_env
source ddp_env/bin/activate
pip install torch torchvision torchaudio
```

## Running the DDP Example

### 1. Interactive Session (for testing)

Request an interactive session with GPUs:
```bash
srun --nodes=2 --ntasks-per-node=2 --gres=gpu:2 --time=02:00:00 --pty /bin/bash
```

Then run the DDP script:
```bash
torchrun --nnodes=2 --nproc_per_node=2 nn_ddp.py
```

### 2. Batch Job Submission

Create a SLURM submission script `submit_ddp.sh`:
```bash
#!/bin/bash
#SBATCH --job-name=ddp_training
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --output=ddp_%j.out
#SBATCH --error=ddp_%j.err

module load python/3.10
module load cuda/11.8
module load pytorch/2.0.1

source ddp_env/bin/activate

export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_PORT=12345

srun python -m torch.distributed.run \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=2 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:29500 \
    nn_ddp.py
```

Submit the job:
```bash
sbatch submit_ddp.sh
```

## Key Features of the DDP Implementation

1. **Multi-GPU Training**: Scales training across multiple GPUs and nodes
2. **Gradient Synchronization**: Automatically averages gradients across processes
3. **Efficient Data Loading**: Uses `DistributedSampler` for efficient data distribution

## Monitoring and Debugging

- Check job status: `squeue -u $USER`
- View output: `tail -f ddp_<jobid>.out`
- View errors: `tail -f ddp_<jobid>.err`
- Cancel job: `scancel <jobid>`

## Best Practices

1. **Batch Size**: Scale your batch size with the number of GPUs
2. **Learning Rate**: Consider scaling the learning rate with the batch size
3. **Checkpointing**: Save model checkpoints from rank 0 only
4. **Logging**: Use `torch.distributed.get_rank() == 0` for single-process logging

## Troubleshooting

- **NCCL Errors**: Ensure all nodes can communicate on the network
- **CUDA Out of Memory**: Reduce batch size or use gradient accumulation
- **Hang during training**: Check for deadlocks in your model's forward/backward pass

## Example Output

```
[Epoch 1/10] Loss: 0.6923 | Acc: 0.5123
[Epoch 2/10] Loss: 0.6812 | Acc: 0.5234
...
```

## Additional Resources

- [PyTorch DDP Documentation](https://pytorch.org/docs/stable/notes/ddp.html)
- [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [SLURM Documentation](https://slurm.schedmd.com/)

## Support

For issues with the HPC system, contact: hpc-support@amplitUDE.edu
For code-related questions, open an issue in this repository.
