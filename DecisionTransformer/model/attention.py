"""
Causal self-attention mechanism
"""
import torch
import torch.nn as nn
import math


class CausalSelfAttention(nn.Module):
    """
    Causal (masked) multi-head self-attention
    
    Ensures that position i can only attend to positions <= i,
    preventing the model from "cheating" by looking at future tokens.
    """
    
    def __init__(self, config):
        """
        Initialize attention layer
        
        Args:
            config: DecisionTransformerConfig instance
        """
        super().__init__()
        assert config.hidden_size % config.num_heads == 0, \
            "hidden_size must be divisible by num_heads"
        
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        
        # Query, Key, Value projections for all heads (combined)
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3)
        
        # Output projection
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Causal mask (lower triangular matrix)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_length * 3, config.max_seq_length * 3)).view(
                1, 1, config.max_seq_length * 3, config.max_seq_length * 3
            )
        )
    
    def forward(self, x, attention_mask=None):
        """
        Forward pass
        
        Args:
            x: (batch_size, seq_length, hidden_size)
            attention_mask: Optional mask for padding (batch_size, seq_length)
        
        Returns:
            output: (batch_size, seq_length, hidden_size)
        """
        B, T, C = x.size()  # batch, sequence length, embedding dim
        
        # Calculate Q, K, V for all heads in batch
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        
        # Reshape for multi-head attention: (B, T, C) -> (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores: Q * K^T / sqrt(d_k)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Apply causal mask (prevent attending to future)
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        
        # Apply padding mask if provided
        if attention_mask is not None:
            att = att.masked_fill(
                attention_mask.unsqueeze(1).unsqueeze(2) == 0, 
                float('-inf')
            )
        
        # Softmax to get attention weights
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Apply attention to values
        y = att @ v  # (B, num_heads, T, head_dim)
        
        # Concatenate heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.proj(y))
        
        return y