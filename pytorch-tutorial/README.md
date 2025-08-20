# PyTorch DDP Quickstart for amplitUDE HPC

This guide provides a quickstart for running distributed PyTorch training using DistributedDataParallel (DDP) on the amplitUDE HPC system.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the DDP Example](#running-the-ddp-example)
  - [Interactive Session](#1-interactive-session-for-testing)
  - [Batch Job Submission](#2-batch-job-submission)
- [Key Features](#key-features-of-the-ddp-implementation)
- [Code Examples](#code-examples)
- [Monitoring and Debugging](#monitoring-and-debugging)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Additional Resources](#additional-resources)
- [Support](#support)

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

## Core DDP Concepts

### Key Terms
- **World Size**: Total number of processes across all nodes (GPUs) participating in the training
- **Rank**: Unique ID assigned to each process (0 to world_size-1)
- **Local Rank**: Process ID within a single node (0 to GPUs_per_node-1)
- **Node**: A physical machine (server) containing one or more GPUs
- **Process Group**: A collection of processes that can communicate with each other

### How DDP Works
1. Each GPU runs a copy of the model (replica)
2. The input data is split across processes (data parallelism)
3. During the backward pass, gradients are averaged across all processes
4. All model replicas stay in sync by having identical weights

## SLURM Concepts

### Key Components
- **srun**: Launches parallel jobs across nodes
- **sbatch**: Submits a batch script for execution
- **squeue**: Views job status
- **scancel**: Terminates a running job

### Resource Management
- `--nodes`: Number of compute nodes to use
- `--ntasks-per-node`: Processes per node (typically equals GPUs per node)
- `--gres`: GPU resources (e.g., gpu:2 means 2 GPUs per node)
- `--time`: Maximum wall time for the job

## Code Examples with Detailed Comments

### 1. Basic DDP Setup
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    # Set the master address and port for process communication
    os.environ['MASTER_ADDR'] = 'localhost'  # Address of the rank 0 process
    os.environ['MASTER_PORT'] = '12355'      # Free port for communication
    
    # Initialize the process group
    # backend='nccl' is optimized for NVIDIA GPUs
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set the default CUDA device for this process
    torch.cuda.set_device(rank)
```

### 2. Model Wrapping and Training Loop
```python
def main(rank, world_size):
    setup(rank, world_size)
    
    # Create model and move to the GPU for this process
    model = NeuralNetwork().to(rank)
    
    # Wrap the model with DDP
    # device_ids=[rank] specifies which GPU this process should use
    ddp_model = DDP(model, device_ids=[rank])
    
    # Create optimizer after DDP wrapper
    optimizer = torch.optim.Adam(ddp_model.parameters())
    
    # Training loop
    for epoch in range(epochs):
        # Set epoch for DistributedSampler
        dataloader.sampler.set_epoch(epoch)
        
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(rank), labels.to(rank)
            
            optimizer.zero_grad()
            outputs = ddp_model(inputs)  # Forward pass
            loss = F.cross_entropy(outputs, labels)
            loss.backward()              # Backward pass (gradients are synchronized here)
            optimizer.step()             # Update weights
    
    cleanup()
```

### 3. SLURM Submission Script Explained
```bash
#!/bin/bash
#SBATCH --job-name=ddp_training     # Name of the job
#SBATCH --nodes=2                   # Use 2 nodes
#SBATCH --ntasks-per-node=2         # Run 2 processes per node (1 per GPU)
#SBATCH --gres=gpu:2                # Request 2 GPUs per node
#SBATCH --time=02:00:00             # Maximum runtime (HH:MM:SS)
#SBATCH --output=ddp_%j.out         # Output file
#SBATCH --error=ddp_%j.err          # Error file

# Load required modules
module load python/3.10 cuda/11.8 pytorch/2.0.1

# Activate virtual environment
source ddp_env/bin/activate

# Set up environment variables for DDP
# MASTER_ADDR: Address of the rank 0 process
# MASTER_PORT: Port for process communication
export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_PORT=12345

# Launch the training script using srun
# --nnodes: Number of nodes
# --nproc_per_node: Processes per node (should match --ntasks-per-node)
# --rdzv_*: Parameters for rendezvous (process synchronization)
srun python -m torch.distributed.run \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=2 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:29500 \
    nn_ddp.py
```

## Understanding the Components

### Process Initialization
1. Each process gets a unique rank (0 to world_size-1)
2. Rank 0 is responsible for model initialization and checkpoint saving
3. All processes must call `init_process_group()` with the same arguments

### Data Loading
- `DistributedSampler` ensures each process gets a unique subset of data
- Set `shuffle=True` for randomness across epochs
- Call `sampler.set_epoch(epoch)` at the beginning of each epoch

### Model Synchronization
- DDP automatically handles gradient synchronization during `loss.backward()`
- All processes maintain identical model parameters
- Only rank 0 needs to save checkpoints

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

## Code Examples

### 1. Basic DDP Setup
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
```

### 2. Model Wrapping
```python
def main(rank, world_size):
    setup(rank, world_size)
    
    # Create model and move to GPU
    model = NeuralNetwork().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    # Create optimizer after DDP wrapper
    optimizer = torch.optim.Adam(ddp_model.parameters())
    
    # Training loop
    for epoch in range(epochs):
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(rank), labels.to(rank)
            
            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
    
    cleanup()
```

### 3. Distributed Data Loading
```python
from torch.utils.data.distributed import DistributedSampler

def prepare_dataloader(dataset, batch_size):
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
```

### 4. Saving and Loading Checkpoints
```python
def save_checkpoint(model, optimizer, epoch, filename):
    if dist.get_rank() == 0:  # Only save from master process
        checkpoint = {
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename, map_location=f'cuda:{dist.get_rank()}')
    model.module.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']
```

### 5. Distributed Evaluation
```python
def evaluate(model, dataloader, device):
    model.eval()
    total = 0
    correct = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Aggregate results across all processes
    total = torch.tensor(total, device=device)
    correct = torch.tensor(correct, device=device)
    
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    
    if dist.get_rank() == 0:
        accuracy = 100 * correct / total
        print(f'Accuracy: {accuracy:.2f}%')
```

### 6. Cleaning Up
```python
def cleanup():
    dist.destroy_process_group()

if __name__ == "__main__":
    import torch.multiprocessing as mp
    
    world_size = 2  # Number of GPUs
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
```

## Common SLURM Commands

1. Submit a job:
   ```bash
   sbatch submit_ddp.sh
   ```

2. Check job status:
   ```bash
   squeue -u $USER
   ```

3. View output:
   ```bash
   tail -f ddp_<jobid>.out
   ```

4. Cancel a job:
   ```bash
   scancel <jobid>
   ```

## Troubleshooting Common Issues

1. **NCCL Errors**:
   - Ensure all nodes can communicate on the network
   - Check firewall settings
   - Verify NCCL is properly installed

2. **CUDA Out of Memory**:
   - Reduce batch size
   - Use gradient accumulation
   - Clear CUDA cache: `torch.cuda.empty_cache()`

3. **Hanging During Training**:
   - Check for deadlocks in model code
   - Ensure all processes execute the same code path
   - Verify all processes call `backward()` the same number of times

## Best Practices

1. **Batch Size**:
   - Scale batch size linearly with the number of GPUs
   - Example: If baseline batch size is 32 on 1 GPU, use 64 for 2 GPUs

2. **Learning Rate**:
   - Linear scaling rule: `lr = base_lr * num_gpus`
   - Consider using learning rate warmup

3. **Checkpointing**:
   - Only save from rank 0 to avoid race conditions
   - Include optimizer state for resuming training

4. **Logging**:
   - Use `if dist.get_rank() == 0:` for single-process logging
   - Consider using TensorBoard with a shared filesystem

## Additional Resources

- [PyTorch DDP Documentation](https://pytorch.org/docs/stable/notes/ddp.html)
- [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [SLURM Documentation](https://slurm.schedmd.com/)

## Support

For issues with the HPC system, contact: hpc-support@amplitUDE.edu  
For code-related questions, open an issue in this repository.
