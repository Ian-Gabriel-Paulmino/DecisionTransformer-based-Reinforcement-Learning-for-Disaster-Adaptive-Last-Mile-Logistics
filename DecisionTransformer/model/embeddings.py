"""
Embedding layer for Decision Transformer
"""
import torch
import torch.nn as nn


class Embeddings(nn.Module):
    """
    Embedding layer for Decision Transformer
    
    Creates embeddings for returns-to-go, states, and actions,
    then interleaves them into a single sequence.
    """
    
    def __init__(self, config):
        """
        Initialize embeddings
        
        Args:
            config: DecisionTransformerConfig instance
        """
        super().__init__()
        self.config = config
        
        # Embed each modality to hidden_size
        self.return_embedding = nn.Linear(1, config.hidden_size)
        self.state_embedding = nn.Linear(config.state_dim, config.hidden_size)
        self.action_embedding = nn.Embedding(config.num_actions, config.hidden_size)
        
        # Positional embedding for sequence position
        self.position_embedding = nn.Embedding(
            config.max_seq_length * 3,  # *3 because we have (R, s, a) per timestep
            config.hidden_size
        )
        
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, returns, states, actions, timesteps):
        """
        Forward pass
        
        Args:
            returns: (batch_size, seq_length, 1)
            states: (batch_size, seq_length, state_dim)
            actions: (batch_size, seq_length)
            timesteps: (batch_size, seq_length)
        
        Returns:
            embeddings: (batch_size, seq_length * 3, hidden_size)
        """
        batch_size, seq_length = returns.shape[0], returns.shape[1]
        
        # Embed each modality
        return_embeddings = self.return_embedding(returns)  # (B, L, H)
        state_embeddings = self.state_embedding(states)     # (B, L, H)
        action_embeddings = self.action_embedding(actions)  # (B, L, H)
        
        # Create interleaved sequence: [R₀, s₀, a₀, R₁, s₁, a₁, ...]
        embeddings = torch.zeros(
            batch_size, seq_length * 3, self.config.hidden_size,
            dtype=return_embeddings.dtype, device=return_embeddings.device
        )
        embeddings[:, 0::3, :] = return_embeddings  # Every 3rd position starting at 0
        embeddings[:, 1::3, :] = state_embeddings   # Every 3rd position starting at 1
        embeddings[:, 2::3, :] = action_embeddings  # Every 3rd position starting at 2
        
        # Add positional embeddings
        position_ids = torch.arange(
            seq_length * 3, device=embeddings.device
        ).unsqueeze(0).expand(batch_size, -1)
        
        position_embeddings = self.position_embedding(position_ids)
        embeddings = embeddings + position_embeddings
        
        # Layer norm and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings