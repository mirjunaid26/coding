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
    
    # Verify data file exists (only on rank 0 to avoid race conditions)
    if rank == 0:
        if not os.path.isfile(args.data_path):
            print(f"Error: Data file not found at {args.data_path}")
            # Signal other processes to exit
            if world_size > 1:
                dist.barrier()
            return
    
    if world_size > 1:
        dist.barrier()  # Wait for rank 0 to check the file
    
    try:
        # Create dataloader first to validate data and get vocab size
        if rank == 0:
            print(f"Loading data from: {args.data_path}")
            print(f"File size: {os.path.getsize(args.data_path) / (1024*1024):.2f} MB")
        
        dataloader, dataset = get_dataloader(
            args.data_path,
            batch_size=args.batch_size,
            block_size=args.block_size,
            num_workers=4
        )
        
        # Update vocab_size from the actual dataset if available
        if hasattr(dataset, 'vocab_size'):
            args.vocab_size = dataset.vocab_size
            if rank == 0:
                print(f"Using vocab size: {args.vocab_size}")
        else:
            if rank == 0:
                print(f"Using default vocab size: {args.vocab_size}")
        
        # Create model and move to GPU
        model = GPT(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_layer=args.n_layer,
            n_head=args.n_head,
            block_size=args.block_size
        ).to(rank)
        
        # Wrap model with DDP
        if world_size > 1:
            model = DDP(model, device_ids=[rank])
        
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        
        # Create TensorBoard writer (only on rank 0)
        if rank == 0:
            os.makedirs(args.log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=args.log_dir)
            print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        
        # Training loop
        model.train()
        total_loss = 0.0
        start_time = time.time()
        
        for step in range(1, args.max_steps + 1):
            try:
                # Get batch of data
                x, y = get_batch(dataloader, rank)
                
                # Forward pass
                _, loss = model(x, y)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                # Log metrics
                total_loss += loss.item()
                
                if step % args.log_interval == 0 and rank == 0:
                    avg_loss = total_loss / args.log_interval
                    elapsed = time.time() - start_time
                    print(f"Step {step:5d} | Loss: {avg_loss:.4f} | "
                          f"Time: {elapsed:.2f}s | "
                          f"Samples/s: {args.log_interval * args.batch_size * world_size / elapsed:.1f}")
                    
                    if writer is not None:
                        writer.add_scalar('train/loss', avg_loss, step)
                        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], step)
                    
                    total_loss = 0.0
                    start_time = time.time()
                
                # Save checkpoint
                if step % args.save_interval == 0 and rank == 0:
                    os.makedirs(args.save_dir, exist_ok=True)
                    checkpoint = {
                        'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'step': step,
                        'config': vars(args)
                    }
                    checkpoint_path = os.path.join(args.save_dir, f"ckpt_step_{step:06d}.pt")
                    torch.save(checkpoint, checkpoint_path)
                    print(f"Saved checkpoint at {checkpoint_path}")
                    
            except Exception as e:
                print(f"Error in training step {step}: {str(e)}")
                if world_size > 1:
                    dist.barrier()
                raise
                
    except Exception as e:
        print(f"Error in training process {rank}: {str(e)}")
        raise
    finally:
        # Cleanup
        if world_size > 1:
            dist.barrier()
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
    parser.add_argument('--vocab_size', type=int, default=50000, 
                       help='vocabulary size (will be overridden by dataset if smaller)')
    
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
