import torch
import torch.nn as nn
from typing import Tuple

class NeuralMemoryModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.input_dim = config['model']['input_dim']
        self.memory_dim = config['model']['memory_dim']
        self.num_layers = config['model']['num_memory_layers']
        self.dropout = config['model']['dropout']
        
        # Projection layers
        self.key_proj = nn.Linear(self.input_dim, self.memory_dim)
        self.value_proj = nn.Linear(self.input_dim, self.memory_dim)
        
        # Memory network
        layers = []
        for i in range(self.num_layers):
            if i == 0:
                layers.append(nn.Linear(self.memory_dim, self.memory_dim))
            else:
                layers.append(nn.Linear(self.memory_dim, self.memory_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(self.dropout))
        self.memory_net = nn.Sequential(*layers)
        
        # Update parameters
        self.theta = nn.Parameter(torch.ones(1))
        self.eta = nn.Parameter(torch.ones(1))
        self.alpha = nn.Parameter(torch.ones(1))
        
        self.register_buffer('surprise', torch.zeros(self.memory_dim))
        
    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        input_size = x.size(2)
        
        # Project input to key and value spaces
        k = self.key_proj(x.reshape(-1, input_size)).reshape(batch_size, seq_len, -1)  # [batch_size, seq_len, memory_dim]
        v = self.value_proj(x.reshape(-1, input_size)).reshape(batch_size, seq_len, -1)  # [batch_size, seq_len, memory_dim]
        
        # Process each sequence element through the memory network
        outputs = []
        for i in range(k.size(1)):
            ki = k[:, i]  # [batch_size, memory_dim]
            
            # Pass through memory network
            pred = ki
            for layer in self.memory_net:
                pred = layer(pred)
            
            outputs.append(pred)
        
        # Stack predictions
        pred = torch.stack(outputs, dim=1)  # [batch_size, seq_len, memory_dim]
        
        # Calculate surprise as L2 distance between prediction and actual value
        surprise = torch.norm(pred - v, dim=-1, keepdim=True)  # [batch_size, seq_len, 1]
        
        return pred, surprise 