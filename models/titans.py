import torch
import torch.nn as nn
from .memory_module import NeuralMemoryModule
from .attention import MultiHeadAttention

class TitansModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.memory = NeuralMemoryModule(config)
        self.attention = MultiHeadAttention(config)
        
        self.input_dim = config['model']['input_dim']
        self.memory_dim = config['model']['memory_dim']
        
        # Output projection
        self.output_proj = nn.Linear(self.memory_dim, self.input_dim)
        
    def forward(self, x: torch.Tensor, memory_tokens: torch.Tensor = None):
        batch_size = x.size(0)
        
        # Get memory output
        memory_out, surprise = self.memory(x)
        
        # If we have memory tokens, use attention
        if memory_tokens is not None:
            # Create attention mask (causal)
            seq_len = memory_tokens.size(1)
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Apply attention
            attended = self.attention(memory_out, memory_tokens, memory_tokens, mask)
            output = attended + memory_out  # Residual connection
        else:
            output = memory_out
            
        return self.output_proj(output), surprise 