# """
# Configuration for Decision Transformer model
# """


# class DecisionTransformerConfig:
#     """
#     Configuration class for Decision Transformer
    
#     Holds all hyperparameters for the model architecture.
#     """
    
#     def __init__(
#         self,
#         state_dim=76,
#         num_actions=20,
#         hidden_size=128,
#         num_layers=6,
#         num_heads=8,
#         dropout=0.1,
#         max_seq_length=60,
#         max_ep_len=20
#     ):
#         """
#         Initialize configuration
        
#         Args:
#             state_dim: Dimension of state vectors (default: 76)
#             num_actions: Number of possible actions (delivery nodes)
#             hidden_size: Dimension of hidden layers (default: 128)
#             num_layers: Number of transformer layers (default: 6)
#             num_heads: Number of attention heads (default: 8)
#             dropout: Dropout probability (default: 0.1)
#             max_seq_length: Maximum sequence length (default: 60)
#             max_ep_len: Maximum episode length (default: 20)
#         """
#         self.state_dim = state_dim
#         self.num_actions = num_actions
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.num_heads = num_heads
#         self.dropout = dropout
#         self.max_seq_length = max_seq_length
#         self.max_ep_len = max_ep_len


"""
Optimized Model Configuration for Limited Data

Key changes from original:
1. Smaller model (fewer parameters to prevent overfitting)
2. More regularization (higher dropout)
3. Better suited for 2K-5K trajectories
"""


class DecisionTransformerConfig:
    """
    Optimized configuration for learning from limited data
    
    Original (too large for <10K samples):
        hidden_size=128, num_layers=6, num_heads=8
        ~800K-1M parameters
    
    Optimized (right-sized for 2K-10K samples):
        hidden_size=64, num_layers=3, num_heads=4
        ~200K-300K parameters
    """
    
    def __init__(
        self,
        state_dim=76,
        num_actions=20,
        hidden_size=64,        # Reduced from 128
        num_layers=3,          # Reduced from 6
        num_heads=4,           # Reduced from 8
        dropout=0.15,          # Increased from 0.1 (more regularization)
        max_seq_length=60,
        max_ep_len=20
    ):
        """
        Initialize optimized configuration
        
        Args:
            state_dim: Dimension of state vectors (76 for your setup)
            num_actions: Number of possible actions (20 delivery nodes)
            hidden_size: Hidden dimension (64 = smaller model)
            num_layers: Number of transformer layers (3 = shallower)
            num_heads: Number of attention heads (4 = fewer heads)
            dropout: Dropout probability (0.15 = more regularization)
            max_seq_length: Maximum sequence length
            max_ep_len: Maximum episode length
        """
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_seq_length = max_seq_length
        self.max_ep_len = max_ep_len
        
        # Validate configuration
        assert hidden_size % num_heads == 0, \
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        
        print(f"Optimized Config Created:")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Layers: {num_layers}")
        print(f"  Heads: {num_heads}")
        print(f"  Dropout: {dropout}")
        print(f"  Estimated parameters: ~{self.estimate_parameters():,}")
    
    def estimate_parameters(self):
        """
        Estimate total number of parameters
        
        Rough calculation:
        - Embeddings: (state_dim + 1 + num_actions) * hidden_size
        - Transformer blocks: num_layers * (4 * hidden_size^2)
        - Output head: hidden_size * num_actions
        """
        embedding_params = (self.state_dim + 1 + self.num_actions) * self.hidden_size
        transformer_params = self.num_layers * (4 * self.hidden_size ** 2)
        output_params = self.hidden_size * self.num_actions
        
        total = embedding_params + transformer_params + output_params
        return int(total)
    


def create_optimized_config_for_data_size(num_trajectories):
    """
    Create appropriately-sized config based on data size
    
    Rule of thumb: ~100 samples per 1K parameters
    
    Args:
        num_trajectories: Number of training trajectories
    
    Returns:
        DecisionTransformerConfig instance
    """
    if num_trajectories < 1000:
        # Very small dataset - tiny model
        return DecisionTransformerConfig(
            hidden_size=32,
            num_layers=2,
            num_heads=2,
            dropout=0.2
        )
    elif num_trajectories < 3000:
        # Small dataset - small model
        return DecisionTransformerConfig(
            hidden_size=64,
            num_layers=3,
            num_heads=4,
            dropout=0.15
        )
    elif num_trajectories < 8000:
        # Medium dataset - medium model
        return DecisionTransformerConfig(
            hidden_size=96,
            num_layers=4,
            num_heads=4,
            dropout=0.12
        )
    else:
        # Large dataset - can use original larger model
        return DecisionTransformerConfig(
            hidden_size=128,
            num_layers=6,
            num_heads=8,
            dropout=0.1
        )
