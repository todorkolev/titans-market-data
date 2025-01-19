import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.memory_dim = config['model']['memory_dim']
        self.num_heads = config['model']['num_attention_heads']
        self.head_dim = self.memory_dim // self.num_heads
        
        self.q_proj = nn.Linear(self.memory_dim, self.memory_dim)
        self.k_proj = nn.Linear(self.memory_dim, self.memory_dim)
        self.v_proj = nn.Linear(self.memory_dim, self.memory_dim)
        self.out_proj = nn.Linear(self.memory_dim, self.memory_dim)
        
        self.dropout = nn.Dropout(config['model']['dropout'])
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        # Project and reshape
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Compute output
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, -1, self.memory_dim)
        
        return self.out_proj(out) 