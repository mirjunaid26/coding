import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

class CharDataset(Dataset):
    def __init__(self, data, block_size):
        chars = sorted(list(set(data)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.block_size = block_size
        self.data = data
        
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        # Grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        # Convert characters to integers
        dix = [self.stoi[s] for s in chunk]
        # Split into input and target
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

def get_dataloader(file_path, batch_size, block_size, num_workers=4):
    # Read the text file
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Create dataset
    dataset = CharDataset(text, block_size)
    
    # Create distributed sampler if DDP is enabled
    sampler = None
    if torch.distributed.is_initialized():
        sampler = DistributedSampler(dataset)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler
    )
    
    return dataloader, dataset

def get_batch(dataloader, device):
    """
    Get a batch of data from the dataloader and move it to the specified device.
    """
    x, y = next(iter(dataloader))
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    return x, y
