"""
Decision Transformer model module
"""
from .config import DecisionTransformerConfig
from .embeddings import Embeddings
from .attention import CausalSelfAttention
from .transformer_block import TransformerBlock
from .decision_transformer import DecisionTransformer
from .config import create_optimized_config_for_data_size

__all__ = [
    'DecisionTransformerConfig',
    'Embeddings',
    'CausalSelfAttention',
    'TransformerBlock',
    'DecisionTransformer',
    'create_optimized_config_for_data_size'
]