"""
Action encoding for Decision Transformer

This module converts delivery node IDs (actions) to indices and vice versa.
"""


class ActionEncoder:
    """
    Encodes actions (delivery node IDs) as indices
    
    The Decision Transformer outputs a probability distribution over actions.
    This class manages the mapping between node IDs and action indices.
    """
    
    def __init__(self, all_delivery_nodes):
        """
        Initialize action encoder
        
        Args:
            all_delivery_nodes: List of all delivery node IDs
        """
        # Sort nodes for consistent ordering
        self.all_delivery_nodes = sorted(all_delivery_nodes)
        
        # Create bidirectional mappings
        self.node_to_idx = {node: idx for idx, node in enumerate(self.all_delivery_nodes)}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        
        # Total number of possible actions
        self.num_actions = len(self.all_delivery_nodes)
    
    def encode_action(self, node_id):
        """
        Convert node ID to action index
        
        Args:
            node_id: Node ID to encode
        
        Returns:
            int: Action index (0 to num_actions-1)
        """
        return self.node_to_idx[node_id]
    
    def decode_action(self, action_idx):
        """
        Convert action index back to node ID
        
        Args:
            action_idx: Action index (0 to num_actions-1)
        
        Returns:
            Node ID corresponding to the action
        """
        return self.idx_to_node[action_idx]