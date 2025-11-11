"""
Main Decision Transformer model
"""
import torch
import torch.nn as nn

from .config import DecisionTransformerConfig
from .embeddings import Embeddings
from .transformer_block import TransformerBlock


class DecisionTransformer(nn.Module):
    """
    Decision Transformer for delivery route optimization
    
    Architecture:
        Input: (returns-to-go, states, actions) tuples
               ↓
        Embeddings (with positional encoding)
               ↓
        N × Transformer Blocks (self-attention + FFN)
               ↓
        Action Prediction Head
               ↓
        Output: Probability distribution over next actions
    """
    
    def __init__(self, config):
        """
        Initialize Decision Transformer
        
        Args:
            config: DecisionTransformerConfig instance
        """
        super().__init__()
        self.config = config
        
        # Embedding layer
        self.embeddings = Embeddings(config)
        
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Final layer normalization
        self.ln_f = nn.LayerNorm(config.hidden_size)
        
        # Action prediction head
        self.action_head = nn.Linear(config.hidden_size, config.num_actions)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """
        Initialize weights using normal distribution
        
        Args:
            module: PyTorch module to initialize
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, returns, states, actions, timesteps, attention_mask=None):
        """
        Forward pass
        
        Args:
            returns: (batch_size, seq_length, 1) - returns-to-go
            states: (batch_size, seq_length, state_dim) - state vectors
            actions: (batch_size, seq_length) - action indices
            timesteps: (batch_size, seq_length) - timestep indices
            attention_mask: (batch_size, seq_length * 3) - padding mask
        
        Returns:
            action_logits: (batch_size, seq_length, num_actions)
        """
        # Get embeddings (interleaved R, s, a sequence)
        x = self.embeddings(returns, states, actions, timesteps)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)
        
        # Final layer normalization
        x = self.ln_f(x)
        
        # Extract action token positions (every 3rd token, starting from index 2)
        # [R₀, s₀, a₀, R₁, s₁, a₁, ...] -> extract a₀, a₁, ...
        action_hidden = x[:, 2::3, :]
        
        # Predict action probabilities
        action_logits = self.action_head(action_hidden)
        
        return action_logits
    
    def get_action(self, returns, states, actions, timesteps, attention_mask=None):
        """
        Get action for the last timestep (for inference)
        
        Args:
            Same as forward()
        
        Returns:
            action_logits: (batch_size, num_actions) - logits for last timestep only
        """
        action_logits = self.forward(returns, states, actions, timesteps, attention_mask)
        return action_logits[:, -1, :]  # Return logits for last timestep