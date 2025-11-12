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


# """
# FIXED: Optimized Model Configuration for Limited Data

# Critical fix: Much more aggressive size reduction for small datasets
# """


# class DecisionTransformerConfig:
#     """
#     Optimized configuration for learning from limited data
#     """
    
#     def __init__(
#         self,
#         state_dim=76,
#         num_actions=20,
#         hidden_size=64,
#         num_layers=3,
#         num_heads=4,
#         dropout=0.15,
#         max_seq_length=60,
#         max_ep_len=20
#     ):
#         """
#         Initialize optimized configuration
        
#         Args:
#             state_dim: Dimension of state vectors (76 for your setup)
#             num_actions: Number of possible actions (20 delivery nodes)
#             hidden_size: Hidden dimension
#             num_layers: Number of transformer layers
#             num_heads: Number of attention heads
#             dropout: Dropout probability
#             max_seq_length: Maximum sequence length
#             max_ep_len: Maximum episode length
#         """
#         self.state_dim = state_dim
#         self.num_actions = num_actions
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.num_heads = num_heads
#         self.dropout = dropout
#         self.max_seq_length = max_seq_length
#         self.max_ep_len = max_ep_len
        
#         # Validate configuration
#         assert hidden_size % num_heads == 0, \
#             f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        
#         print(f"Config Created:")
#         print(f"  Hidden size: {hidden_size}")
#         print(f"  Layers: {num_layers}")
#         print(f"  Heads: {num_heads}")
#         print(f"  Dropout: {dropout}")
#         print(f"  Estimated parameters: ~{self.estimate_parameters():,}")
    
#     def estimate_parameters(self):
#         """
#         Estimate total number of parameters
        
#         Rough calculation:
#         - Embeddings: (state_dim + 1 + num_actions) * hidden_size
#         - Transformer blocks: num_layers * (4 * hidden_size^2)
#         - Output head: hidden_size * num_actions
#         """
#         embedding_params = (self.state_dim + 1 + self.num_actions) * self.hidden_size
#         transformer_params = self.num_layers * (4 * self.hidden_size ** 2)
#         output_params = self.hidden_size * self.num_actions
        
#         total = embedding_params + transformer_params + output_params
#         return int(total)


# def create_optimized_config_for_data_size(num_trajectories):
#     """
#     FIXED: Create appropriately-sized config based on data size
    
#     Rule of thumb: Aim for 0.1-0.2 samples per parameter ratio
#     (10-20 samples per 100 parameters)
    
#     Args:
#         num_trajectories: Number of training trajectories
    
#     Returns:
#         DecisionTransformerConfig instance
#     """
#     print(f"\n🔧 Selecting model size for {num_trajectories} trajectories...")
    
#     if num_trajectories < 1000:
#         # Very small dataset - tiny model (~10K params)
#         print("   📉 Very small dataset detected")
#         config = DecisionTransformerConfig(
#             hidden_size=24,
#             num_layers=2,
#             num_heads=2,
#             dropout=0.25
#         )
#     elif num_trajectories < 3000:
#         # Small dataset - small model (~25K params)
#         print("   📉 Small dataset detected")
#         config = DecisionTransformerConfig(
#             hidden_size=32,
#             num_layers=2,
#             num_heads=4,
#             dropout=0.2
#         )
#     elif num_trajectories < 8000:
#         # Medium-small dataset - medium-small model (~50-60K params)
#         print("   📊 Medium-small dataset detected")
#         config = DecisionTransformerConfig(
#             hidden_size=48,    # REDUCED from 96
#             num_layers=2,      # REDUCED from 4
#             num_heads=4,
#             dropout=0.2        # INCREASED from 0.12
#         )
#     elif num_trajectories < 15000:
#         # Medium dataset - medium model (~100-150K params)
#         print("   📊 Medium dataset detected")
#         config = DecisionTransformerConfig(
#             hidden_size=64,
#             num_layers=3,
#             num_heads=4,
#             dropout=0.15
#         )
#     else:
#         # Large dataset - can use larger model (~250K+ params)
#         print("   📈 Large dataset detected")
#         config = DecisionTransformerConfig(
#             hidden_size=96,
#             num_layers=4,
#             num_heads=4,
#             dropout=0.12
#         )
    
#     # Calculate and display ratio
#     estimated_params = config.estimate_parameters()
#     ratio = num_trajectories / estimated_params
    
#     print(f"   Samples-to-parameters ratio: {ratio:.3f}")
    
#     if ratio < 0.05:
#         print(f"   ⚠️  CRITICAL: Very low ratio! Expect severe overfitting")
#         print(f"   💡 Recommendation: Collect more data or reduce model further")
#     elif ratio < 0.1:
#         print(f"   ⚠️  WARNING: Low ratio. Monitor for overfitting closely")
#         print(f"   💡 Recommendation: Use early stopping and high regularization")
#     elif ratio < 0.2:
#         print(f"   ✓ Acceptable ratio. Should train reasonably well")
#     else:
#         print(f"   ✓ Good ratio. Training should be stable")
    
#     return config




# """
# IMPROVED Model Configuration - Better sized for routing task
# """


# class DecisionTransformerConfig:
#     """
#     Configuration optimized for delivery routing
#     """
    
#     def __init__(
#         self,
#         state_dim=76,
#         num_actions=20,
#         hidden_size=64,
#         num_layers=3,
#         num_heads=4,
#         dropout=0.15,
#         max_seq_length=60,
#         max_ep_len=20
#     ):
#         self.state_dim = state_dim
#         self.num_actions = num_actions
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.num_heads = num_heads
#         self.dropout = dropout
#         self.max_seq_length = max_seq_length
#         self.max_ep_len = max_ep_len
        
#         assert hidden_size % num_heads == 0, \
#             f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        
#         print(f"Config Created:")
#         print(f"  Hidden size: {hidden_size}")
#         print(f"  Layers: {num_layers}")
#         print(f"  Heads: {num_heads}")
#         print(f"  Dropout: {dropout}")
#         print(f"  Estimated parameters: ~{self.estimate_parameters():,}")
    
#     def estimate_parameters(self):
#         embedding_params = (self.state_dim + 1 + self.num_actions) * self.hidden_size
#         transformer_params = self.num_layers * (4 * self.hidden_size ** 2)
#         output_params = self.hidden_size * self.num_actions
        
#         total = embedding_params + transformer_params + output_params
#         return int(total)


# def create_optimized_config_for_data_size(num_trajectories):
#     """
#     IMPROVED: Larger models that can actually learn routing
    
#     Previous config was too small (48 hidden, 2 layers = 24K params)
#     This caused poor performance because model couldn't learn patterns
    
#     New strategy: Use reasonable model sizes that can learn
#     """
#     print(f"\n🔧 Selecting model size for {num_trajectories} trajectories...")
    
#     if num_trajectories < 3000:
#         # Small dataset - but still need enough capacity
#         print("   📉 Small dataset detected")
#         config = DecisionTransformerConfig(
#             hidden_size=64,   # Up from 32/48
#             num_layers=3,      # Up from 2
#             num_heads=4,
#             dropout=0.15
#         )
#     elif num_trajectories < 10000:
#         # Medium dataset - good model
#         print("   📊 Medium dataset detected")
#         config = DecisionTransformerConfig(
#             hidden_size=96,    # Up from 48
#             num_layers=4,      # Up from 2
#             num_heads=4,
#             dropout=0.12
#         )
#     elif num_trajectories < 20000:
#         # Large dataset - can use full capacity
#         print("   📈 Large dataset detected")
#         config = DecisionTransformerConfig(
#             hidden_size=128,
#             num_layers=6,
#             num_heads=8,
#             dropout=0.1
#         )
#     else:
#         # Very large dataset - maximum capacity
#         print("   📈📈 Very large dataset detected")
#         config = DecisionTransformerConfig(
#             hidden_size=256,
#             num_layers=8,
#             num_heads=8,
#             dropout=0.1
#         )
    
#     # Calculate ratio
#     estimated_params = config.estimate_parameters()
#     ratio = num_trajectories / estimated_params
    
#     print(f"   Samples-to-parameters ratio: {ratio:.3f}")
    
#     if ratio < 0.05:
#         print(f"   ⚠️  WARNING: Low ratio - model may overfit")
#         print(f"   💡 Recommendation: Use high dropout and early stopping")
#     elif ratio < 0.1:
#         print(f"   ✓ Acceptable ratio with proper regularization")
#     else:
#         print(f"   ✓✓ Good ratio - should train well")
    
#     return config





"""
CORRECTED Model Configuration
Properly sized for 10K training samples with limited weather 4/5 data
"""


class DecisionTransformerConfig:
    """Configuration optimized for delivery routing"""
    
    def __init__(
        self,
        state_dim=76,
        num_actions=20,
        hidden_size=64,
        num_layers=3,
        num_heads=4,
        dropout=0.15,
        max_seq_length=60,
        max_ep_len=20
    ):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_seq_length = max_seq_length
        self.max_ep_len = max_ep_len
        
        assert hidden_size % num_heads == 0, \
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        
        print(f"Config Created:")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Layers: {num_layers}")
        print(f"  Heads: {num_heads}")
        print(f"  Dropout: {dropout}")
        print(f"  Estimated parameters: ~{self.estimate_parameters():,}")
    
    def estimate_parameters(self):
        embedding_params = (self.state_dim + 1 + self.num_actions) * self.hidden_size
        transformer_params = self.num_layers * (4 * self.hidden_size ** 2)
        output_params = self.hidden_size * self.num_actions
        
        total = embedding_params + transformer_params + output_params
        return int(total)


def create_optimized_config_for_data_size(num_trajectories):
    """
    CORRECTED: Properly sized models with high regularization
    
    Key insight: With Weather 4/5 having mostly failures, we need:
    - Smaller models to avoid overfitting on Weather 1-3
    - Higher dropout for robustness
    - Models that generalize well
    
    Args:
        num_trajectories: Number of TRAINING trajectories (not total)
    """
    print(f"\n🔧 Selecting model size for {num_trajectories} trajectories...")
    
    if num_trajectories < 5000:
        # Small dataset
        print("   📉 Small dataset detected")
        config = DecisionTransformerConfig(
            hidden_size=64,
            num_layers=3,
            num_heads=4,
            dropout=0.2  # High dropout for generalization
        )
    elif num_trajectories < 12000:
        # Medium dataset (YOUR CASE: 10,594)
        print("   📊 Medium dataset detected")
        config = DecisionTransformerConfig(
            hidden_size=96,     # Reduced from 128
            num_layers=4,       # Reduced from 6
            num_heads=4,        # Reduced from 8
            dropout=0.15        # Increased from 0.1
        )
    elif num_trajectories < 20000:
        # Large dataset
        print("   📈 Large dataset detected")
        config = DecisionTransformerConfig(
            hidden_size=128,
            num_layers=5,
            num_heads=8,
            dropout=0.12
        )
    else:
        # Very large dataset
        print("   📈📈 Very large dataset detected")
        config = DecisionTransformerConfig(
            hidden_size=128,
            num_layers=6,
            num_heads=8,
            dropout=0.1
        )
    
    # Calculate ratio
    estimated_params = config.estimate_parameters()
    ratio = num_trajectories / estimated_params
    
    print(f"   Samples-to-parameters ratio: {ratio:.3f}")
    
    if ratio < 0.05:
        print(f"   ⚠️  CRITICAL: Very low ratio - high overfitting risk")
        print(f"   💡 URGENT: Use smaller model or collect more data")
    elif ratio < 0.1:
        print(f"   ⚠️  WARNING: Low ratio - use high dropout")
        print(f"   💡 Recommendation: Monitor validation loss closely")
    elif ratio < 0.2:
        print(f"   ✓ Acceptable ratio with regularization")
    else:
        print(f"   ✓✓ Good ratio - should train well")
    
    return config