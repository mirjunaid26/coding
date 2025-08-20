import torch
import argparse
from minigpt import GPT

def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Create model with saved config
    model = GPT(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_layer=config['n_layer'],
        n_head=config['n_head'],
        block_size=config['block_size']
    ).to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, config

def encode_text(text, stoi):
    """Encode text using the vocabulary."""
    return [stoi[ch] for ch in text if ch in stoi]

def decode_tokens(tokens, itos):
    """Decode tokens to text."""
    return ''.join([itos[t] for t in tokens])

def main():
    parser = argparse.ArgumentParser(description='Generate text using a trained GPT model')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to model checkpoint')
    parser.add_argument('--prompt', type=str, default='\n', help='starting prompt')
    parser.add_argument('--max_tokens', type=int, default=1000, help='maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='sampling temperature')
    parser.add_argument('--top_k', type=int, default=40, help='top-k sampling')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model and config
    model, config = load_model(args.checkpoint, device)
    
    # Load vocabulary (you'll need to modify this based on your data loading)
    # For now, we'll assume a simple character-level vocabulary
    with open(config['data_path'], 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Create vocabulary (this should match your training setup)
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    # Encode the prompt
    prompt_tokens = encode_text(args.prompt, stoi)
    x = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    
    # Generate text
    print(args.prompt, end='', flush=True)
    with torch.no_grad():
        y = model.module.generate(
            x,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k
        )
        
        # Decode and print the generated text
        generated = y[0].tolist()
        print(decode_tokens(generated[len(prompt_tokens):], itos))

if __name__ == "__main__":
    main()
