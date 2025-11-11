"""
Single transformer block
"""
import torch.nn as nn


class TransformerBlock(nn.Module):
    """
    Single transformer block with causal self-attention and feed-forward network
    
    Structure:
        x -> LayerNorm -> Attention -> + -> LayerNorm -> FFN -> + -> output
        |___________________________|    |____________________|
                (residual)                    (residual)
    """
    
    def __init__(self, config):
        """
        Initialize transformer block
        
        Args:
            config: DecisionTransformerConfig instance
        """
        super().__init__()
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        
        # Causal self-attention
        from .attention import CausalSelfAttention
        self.attn = CausalSelfAttention(config)
        
        # Feed-forward network (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, x, attention_mask=None):
        """
        Forward pass
        
        Args:
            x: (batch_size, seq_length, hidden_size)
            attention_mask: Optional padding mask
        
        Returns:
            output: (batch_size, seq_length, hidden_size)
        """
        # Attention block with residual connection
        x = x + self.attn(self.ln1(x), attention_mask)
        
        # Feed-forward block with residual connection
        x = x + self.mlp(self.ln2(x))
        
        return x