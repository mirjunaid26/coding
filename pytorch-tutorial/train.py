import os
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import SimpleCNN

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def get_dataloaders(batch_size, rank, world_size):
    # Data augmentation and normalization for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Download CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    
    # Create distributed samplers
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=4, pin_memory=True)
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)
    
    return train_loader, test_loader

def train(rank, world_size, args):
    setup(rank, world_size)
    
    # Initialize model and move to GPU
    model = SimpleCNN(num_classes=10).to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Get data loaders
    train_loader, test_loader = get_dataloaders(args.batch_size, rank, world_size)
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_loader.sampler.set_epoch(epoch)
        
        if rank == 0:
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        else:
            pbar = train_loader
        
        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(rank), target.to(rank)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if rank == 0 and batch_idx % 100 == 0:
                pbar.set_description(f'Epoch {epoch+1}/{args.epochs} Loss: {loss.item():.4f}')
        
        # Update learning rate
        scheduler.step()
        
        # Validate on rank 0
        if rank == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(rank), target.to(rank)
                    outputs = model(data)
                    _, predicted = torch.max(outputs.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            
            accuracy = 100 * correct / total
            print(f'\nTest Accuracy: {accuracy:.2f}%\n')
    
    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch DDP CIFAR-10 Training')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--world_size', type=int, default=4, help='Number of GPUs')
    args = parser.parse_args()
    
    # Set the seed for reproducibility
    torch.manual_seed(42)
    
    # Start distributed training
    mp.spawn(train, args=(args.world_size, args), nprocs=args.world_size, join=True)
