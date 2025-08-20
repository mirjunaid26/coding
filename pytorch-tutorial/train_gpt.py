import os
import time
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from minigpt import GPT
from data import get_dataloader, get_batch

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, args):
    print(f"Running DDP on rank {rank}.")
    setup(rank, world_size)
    
    # Create model and move to GPU
    model = GPT(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layer=args.n_layer,
        n_head=args.n_head,
        block_size=args.block_size
    ).to(rank)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank])
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Create dataloader
    dataloader, dataset = get_dataloader(
        args.data_path,
        batch_size=args.batch_size,
        block_size=args.block_size,
        num_workers=4
    )
    
    # Create TensorBoard writer (only on rank 0)
    if rank == 0:
        writer = SummaryWriter(log_dir=args.log_dir)
    
    # Training loop
    model.train()
    total_loss = 0.0
    start_time = time.time()
    
    for step in range(1, args.max_steps + 1):
        # Get batch of data
        x, y = get_batch(dataloader, rank)
        
        # Forward pass
        _, loss = model(x, y)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Log metrics
        total_loss += loss.item()
        
        if step % args.log_interval == 0 and rank == 0:
            avg_loss = total_loss / args.log_interval
            print(f"Step {step:5d} | Loss: {avg_loss:.4f} | "
                  f"Time: {(time.time() - start_time):.2f}s")
            writer.add_scalar('train/loss', avg_loss, step)
            total_loss = 0.0
        
        # Save checkpoint
        if step % args.save_interval == 0 and rank == 0:
            checkpoint = {
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': step,
                'config': vars(args)
            }
            torch.save(checkpoint, f"{args.save_dir}/ckpt_step_{step:06d}.pt")
            print(f"Saved checkpoint at step {step}")
    
    # Cleanup
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GPT with DDP')
    
    # Model parameters
    parser.add_argument('--d_model', type=int, default=768, help='model dimension')
    parser.add_argument('--n_layer', type=int, default=12, help='number of layers')
    parser.add_argument('--n_head', type=int, default=12, help='number of attention heads')
    parser.add_argument('--block_size', type=int, default=1024, help='context length')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help='batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=6e-4, help='learning rate')
    parser.add_argument('--max_steps', type=int, default=100000, help='number of training steps')
    parser.add_argument('--log_interval', type=int, default=10, help='log interval')
    parser.add_argument('--save_interval', type=int, default=1000, help='save interval')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, required=True, help='path to training data')
    parser.add_argument('--vocab_size', type=int, default=50000, help='vocabulary size')
    
    # Output parameters
    parser.add_argument('--log_dir', type=str, default='logs', help='directory for TensorBoard logs')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='directory to save checkpoints')
    
    # DDP parameters
    parser.add_argument('--world_size', type=int, default=4, help='number of GPUs to use')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Launch DDP processes
    mp.spawn(
        train,
        args=(args.world_size, args),
        nprocs=args.world_size,
        join=True
    )
