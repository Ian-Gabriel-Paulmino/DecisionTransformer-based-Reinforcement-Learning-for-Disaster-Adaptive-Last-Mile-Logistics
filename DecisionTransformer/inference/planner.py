"""
Route planning using trained Decision Transformer
"""
import torch
import pickle
import networkx as nx

from DecisionTransformer.model.decision_transformer import DecisionTransformer
from DecisionTransformer.model.config import DecisionTransformerConfig


class DecisionTransformerPlanner:
    """
    Route planner using trained Decision Transformer
    
    Performs autoregressive generation to produce delivery routes.
    """
    
    def __init__(self, model_path, encoders_path, device=None):
        """
        Initialize planner
        
        Args:
            model_path: Path to trained model checkpoint (.pt file)
            encoders_path: Path to saved encoders (.pkl file)
            device: torch device (default: auto-detect)
        """
        # Load encoders
        print(f"Loading encoders from {encoders_path}...")
        with open(encoders_path, 'rb') as f:
            encoders = pickle.load(f)
        
        self.state_encoder = encoders['state_encoder']
        self.action_encoder = encoders['action_encoder']
        
        # Load model
        print(f"Loading model from {model_path}...")
        # checkpoint = torch.load(model_path, map_location='cpu')
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        config_dict = checkpoint['config']
        
        # Recreate config
        self.config = DecisionTransformerConfig(**config_dict)
        
        # Create and load model
        self.model = DecisionTransformer(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Loaded Decision Transformer (device: {self.device})")
    
    def plan_route(self, graph, start_node, delivery_nodes, rain_intensity, 
                   target_return=-30.0, max_steps=20, temperature=0.0):
        """
        Generate delivery route using Decision Transformer
        
        Args:
            graph: NetworkX graph
            start_node: Starting hub node
            delivery_nodes: List of delivery nodes to visit
            rain_intensity: Current weather (1-5)
            target_return: Desired return (negative time in minutes)
            max_steps: Maximum planning steps
            temperature: Sampling temperature (0.0 = greedy, >0 = stochastic)
        
        Returns:
            tuple: (route, estimated_time)
                - route: List of nodes in delivery order
                - estimated_time: Estimated delivery time
        """
        # Initialize
        current_node = start_node
        remaining_nodes = set(delivery_nodes)
        route = [start_node]
        
        # Update state encoder's graph reference
        self.state_encoder.graph = graph
        
        # Sequence buffers
        returns_sequence = []
        states_sequence = []
        actions_sequence = []
        timesteps_sequence = []
        
        estimated_time = 0
        
        with torch.no_grad():
            for step in range(max_steps):
                if not remaining_nodes:
                    break
                
                # Create current state
                state = {
                    'current_node': current_node,
                    'remaining_nodes': sorted(list(remaining_nodes)),
                    'rain_intensity': rain_intensity,
                    'position_in_route': step,
                    'total_nodes': len(delivery_nodes),
                    'disaster_context': self._get_disaster_context(graph, current_node)
                }
                
                # Encode state
                state_vector = self.state_encoder.encode_state(state)
                
                # Add to sequences
                returns_sequence.append(target_return)
                states_sequence.append(state_vector)
                timesteps_sequence.append(step)
                
                # Prepare model inputs (add batch dimension)
                returns_tensor = torch.tensor(
                    returns_sequence, dtype=torch.float32
                ).unsqueeze(0).unsqueeze(-1).to(self.device)

                states_tensor = torch.tensor(
                    states_sequence, dtype=torch.float32
                ).unsqueeze(0).to(self.device)

                # Fix: Ensure actions tensor matches the sequence length
                if actions_sequence:
                    actions_tensor = torch.tensor(
                        actions_sequence, dtype=torch.long
                    ).unsqueeze(0).to(self.device)
                    
                    # Pad with zeros if actions is shorter than states
                    if len(actions_sequence) < len(states_sequence):
                        padding_length = len(states_sequence) - len(actions_sequence)
                        padding = torch.zeros(1, padding_length, dtype=torch.long).to(self.device)
                        actions_tensor = torch.cat([actions_tensor, padding], dim=1)
                else:
                    # First step: create dummy actions matching states length
                    actions_tensor = torch.zeros(
                        1, len(states_sequence), dtype=torch.long
                    ).to(self.device)

                timesteps_tensor = torch.tensor(
                    timesteps_sequence, dtype=torch.long
                ).unsqueeze(0).to(self.device)
                
                # Forward pass
                action_logits = self.model(
                    returns_tensor,
                    states_tensor,
                    actions_tensor,
                    timesteps_tensor
                )
                
                # Get logits for current step
                current_logits = action_logits[0, -1, :]
                
                # Mask out already visited nodes
                mask = self._create_action_mask(remaining_nodes)
                current_logits = current_logits + mask
                
                # Sample action
                if temperature == 0.0:
                    # Greedy selection
                    action_idx = torch.argmax(current_logits).item()
                else:
                    # Sample with temperature
                    probs = torch.softmax(current_logits / temperature, dim=0)
                    action_idx = torch.multinomial(probs, 1).item()
                
                # Decode action
                next_node = self.action_encoder.decode_action(action_idx)
                
                # Estimate time cost for this step
                estimated_step_time = self._estimate_time_between_nodes(
                    graph, current_node, next_node, rain_intensity
                )
                estimated_time += abs(estimated_step_time)
                
                # Update sequences
                actions_sequence.append(action_idx)
                route.append(next_node)
                remaining_nodes.remove(next_node)
                
                # Update target return
                target_return += estimated_step_time  # Add back time spent (both negative)
                
                current_node = next_node
        
        return route, estimated_time
    
    def _create_action_mask(self, available_nodes):
        """
        Create mask to prevent selecting already visited nodes
        
        Args:
            available_nodes: Set of nodes that can still be visited
        
        Returns:
            mask: Tensor with -inf for unavailable actions, 0 for available
        """
        mask = torch.full(
            (self.action_encoder.num_actions,),
            float('-inf'),
            device=self.device
        )
        
        for node in available_nodes:
            if node in self.action_encoder.node_to_idx:
                idx = self.action_encoder.node_to_idx[node]
                mask[idx] = 0.0
        
        return mask
    
    def _get_disaster_context(self, graph, node, radius=3):
        """
        Get disaster information near a node
        
        Args:
            graph: NetworkX graph
            node: Node ID
            radius: Search radius in hops
        
        Returns:
            dict with 'floods_nearby' and 'landslides_nearby' counts
        """
        try:
            nearby_nodes = nx.single_source_shortest_path_length(
                graph, node, cutoff=radius
            ).keys()
        except:
            nearby_nodes = [node]
        
        floods_nearby = 0
        landslides_nearby = 0
        
        for u in nearby_nodes:
            for v in graph.neighbors(u):
                if graph.has_edge(u, v):
                    edge_data = graph.edges[u, v]
                    if edge_data.get('currently_flooded', False):
                        floods_nearby += 1
                    if edge_data.get('currently_landslide', False):
                        landslides_nearby += 1
        
        return {
            'floods_nearby': floods_nearby,
            'landslides_nearby': landslides_nearby
        }
    
    def _estimate_time_between_nodes(self, graph, from_node, to_node, 
                                     rain_intensity, base_speed=500):
        """
        Estimate travel time between two nodes
        
        Args:
            graph: NetworkX graph
            from_node: Source node
            to_node: Target node
            rain_intensity: Weather severity (1-5)
            base_speed: Base speed in m/min
        
        Returns:
            Estimated time (negative for reward)
        """
        try:
            # Get shortest path
            path = nx.shortest_path(graph, from_node, to_node, weight='length')
            distance = sum(
                graph.edges[path[i], path[i+1]].get('length', 0)
                for i in range(len(path) - 1)
            )
            
            # Apply rain slowdown
            rain_effects = {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.4, 5: 0.2}
            speed_multiplier = rain_effects.get(rain_intensity, 0.6)
            
            adjusted_speed = base_speed * speed_multiplier
            time_minutes = distance / adjusted_speed
            
            return -time_minutes  # Negative for reward
        except:
            return -10.0  # Default penalty