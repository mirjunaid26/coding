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
        dix = [self.stoi.get(s, 0) for s in chunk]  # Use .get() with default for robustness
        # Split into input and target
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

def get_dataloader(file_path, batch_size, block_size, num_workers=4):
    # Convert to absolute path for better error messages
    file_path = os.path.abspath(file_path)
    
    # Check if file exists and is accessible
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Training data file not found at: {file_path}")
    
    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"No read permissions for file: {file_path}")
    
    try:
        # Read the text file with explicit encoding
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        if not text.strip():
            raise ValueError(f"File is empty: {file_path}")
            
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
            sampler=sampler,
            drop_last=True  # Drop last incomplete batch
        )
        
        return dataloader, dataset
        
    except UnicodeDecodeError:
        raise ValueError(f"File is not UTF-8 encoded: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading data from {file_path}: {str(e)}")

def get_batch(dataloader, device):
    """
    Get a batch of data from the dataloader and move it to the specified device.
    """
    try:
        x, y = next(iter(dataloader))
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    except StopIteration:
        raise RuntimeError("Dataloader is empty. Check your data path and batch size.")
    except Exception as e:
        raise RuntimeError(f"Error getting batch: {str(e)}")
