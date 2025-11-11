"""
State encoding for Decision Transformer

This module converts delivery states (current node, remaining nodes, weather, etc.)
into fixed-size numerical vectors that the transformer can process.
"""
import numpy as np
import networkx as nx


class StateEncoder:
    """
    Encodes delivery states into fixed-size vectors
    
    A state contains:
    - Current node position
    - Remaining delivery nodes
    - Rain intensity
    - Progress in route
    - Nearby disaster information
    
    Output: 76-dimensional vector
    """
    
    def __init__(self, graph, node_embedding_dim=64):
        """
        Initialize state encoder
        
        Args:
            graph: NetworkX graph of the road network
            node_embedding_dim: Dimension of node embeddings (default: 64)
        """
        self.graph = graph
        self.node_embedding_dim = node_embedding_dim
        
        # Pre-compute node embeddings using simple graph features
        self.node_embeddings = self._compute_node_embeddings()
        
        # Total state dimension: 64 (node) + 1 (rain) + 1 (progress) + 10 (disasters) = 76
        self.state_dim = node_embedding_dim + 1 + 1 + 10
    
    def _compute_node_embeddings(self):
        """
        Compute embeddings for all nodes in the graph
        
        Uses simple features:
        - Normalized x, y coordinates
        - Degree centrality
        - Padded to node_embedding_dim
        
        Returns:
            dict: node_id -> embedding vector
        """
        embeddings = {}
        
        # Compute centrality measures
        degree_centrality = nx.degree_centrality(self.graph)
        
        # Get coordinate ranges for normalization
        all_x = [data['x'] for _, data in self.graph.nodes(data=True)]
        all_y = [data['y'] for _, data in self.graph.nodes(data=True)]
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        
        for node, data in self.graph.nodes(data=True):
            # Normalize coordinates to [0, 1]
            norm_x = (data['x'] - x_min) / (x_max - x_min + 1e-8)
            norm_y = (data['y'] - y_min) / (y_max - y_min + 1e-8)
            degree = degree_centrality[node]
            
            # Create embedding: [x, y, degree, zeros...]
            # Pad with zeros to reach desired dimension
            embedding = np.array([norm_x, norm_y, degree] + 
                                [0.0] * (self.node_embedding_dim - 3))
            embeddings[node] = embedding
        
        return embeddings
    
    def encode_state(self, state):
        """
        Encode a state dictionary into a vector
        
        Args:
            state (dict): State with keys:
                - current_node: Node ID
                - remaining_nodes: List of unvisited node IDs
                - rain_intensity: 1-5 scale
                - position_in_route: Current step number
                - total_nodes: Total number of deliveries
                - disaster_context: Dict with 'floods_nearby', 'landslides_nearby'
        
        Returns:
            np.array: 76-dimensional state vector
        """
        # 1. Current node embedding (64-dim)
        current_node_emb = self.node_embeddings[state['current_node']]
        
        # 2. Rain intensity (scaled to [0, 1])
        rain_feature = np.array([state['rain_intensity'] / 5.0])
        
        # 3. Route progress (scaled to [0, 1])
        progress = state['position_in_route'] / max(state['total_nodes'], 1)
        progress_feature = np.array([progress])
        
        # 4. Disaster context (10-dim)
        disaster_context = state.get('disaster_context', {
            'floods_nearby': 0, 
            'landslides_nearby': 0
        })
        
        # Normalize disaster counts (typical max around 20)
        floods_norm = min(disaster_context['floods_nearby'] / 20.0, 1.0)
        landslides_norm = min(disaster_context['landslides_nearby'] / 20.0, 1.0)
        
        # Create disaster feature vector with padding for future features
        disaster_features = np.array([
            floods_norm,
            landslides_norm,
            0, 0, 0, 0, 0, 0, 0, 0   # Reserved for future features
        ])
        
        # Concatenate all features
        state_vector = np.concatenate([
            current_node_emb,    # 64-dim
            rain_feature,        # 1-dim
            progress_feature,    # 1-dim
            disaster_features    # 10-dim
        ])  # Total: 76-dim
        
        return state_vector.astype(np.float32)