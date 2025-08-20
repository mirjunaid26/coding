import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_k = d_model // n_head
        
        # Key, Query, Value projections
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        
        # Causal mask to ensure attention is only applied to the left in the input sequence
        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(1024, 1024)).view(1, 1, 1024, 1024)
        )
        
    def forward(self, x):
        B, T, C = x.size()
        
        # Calculate query, key, values for all heads
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.d_model * self.n_head, dim=2)
        
        # Reshape to (B, n_head, T, d_k)
        q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_k))
        
        # Apply causal mask
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        # Apply attention to values
        y = att @ v
        
        # Reshape and project back
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=4*1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_head)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model)
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model=768, n_layer=12, n_head=12, block_size=1024):
        super().__init__()
        self.block_size = block_size
        
        # Token and position embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, d_model))
        
        # Transformer blocks
        self.blocks = nn.Sequential(*[Block(d_model, n_head) for _ in range(n_layer)])
        
        # Output head
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.block_size, f"Sequence length {T} exceeds block size {self.block_size}"
        
        # Forward pass
        token_embeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:, :T, :]
        x = token_embeddings + position_embeddings
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        
        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1),
                ignore_index=-1
            )
            
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens given a context.
        """
        for _ in range(max_new_tokens):
            # Crop idx to last block_size tokens
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            # Get predictions
            logits, _ = self(idx_cond)
            
            # Get logits for the last token
            logits = logits[:, -1, :] / temperature
            
            # Optionally crop logits to top-k
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # Get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx
