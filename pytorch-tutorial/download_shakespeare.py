import os
import requests
import numpy as np
from pathlib import Path

def download_file(url, filename):
    """Download a file from a URL and save it locally."""
    if os.path.exists(filename):
        print(f"File {filename} already exists, skipping download.")
        return
        
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save the file
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded {filename}")

def prepare_shakespeare():
    """Download and prepare the Shakespeare dataset."""
    # URLs for the dataset
    base_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/"
    input_file = "input.txt"
    url = f"{base_url}{input_file}"
    
    # Local paths
    data_dir = "data/shakespeare"
    os.makedirs(data_dir, exist_ok=True)
    input_file_path = os.path.join(data_dir, input_file)
    
    # Download the dataset
    download_file(url, input_file_path)
    
    # Read the data
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    
    print(f"Length of dataset in characters: {len(data):,}")
    
    # Split into train and validation
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]
    
    # Save the splits
    train_file = os.path.join(data_dir, 'train.txt')
    val_file = os.path.join(data_dir, 'val.txt')
    
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write(train_data)
    with open(val_file, 'w', encoding='utf-8') as f:
        f.write(val_data)
    
    print(f"Training set size: {len(train_data):,} characters")
    print(f"Validation set size: {len(val_data):,} characters")
    print(f"Data saved to {os.path.abspath(data_dir)}/")
    print("\nSample from training data:")
    print("-" * 50)
    print(train_data[:500])
    print("-" * 50)

if __name__ == "__main__":
    prepare_shakespeare()
