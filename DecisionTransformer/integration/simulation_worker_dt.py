"""
SimulationWorker with Decision Transformer integration
"""
import networkx as nx
import random


def rain_intensity_effects(rain_intensity):
    """Get disaster activation parameters based on rain level"""
    rain_effects = {
        1: {"flood_activation": 0.10, "landslide_activation": 0.00, "speed_multiplier": 1.0},
        2: {"flood_activation": 0.30, "landslide_activation": 0.05, "speed_multiplier": 0.8},
        3: {"flood_activation": 0.60, "landslide_activation": 0.15, "speed_multiplier": 0.6},
        4: {"flood_activation": 0.90, "landslide_activation": 0.30, "speed_multiplier": 0.4},
        5: {"flood_activation": 1.00, "landslide_activation": 1.00, "speed_multiplier": 0.0}
    }
    return rain_effects.get(rain_intensity, rain_effects[1])


class MonteCarloSimulationWorkerWithDT:
    """
    Enhanced SimulationWorker that can use Decision Transformer for routing
    
    Falls back to NNA if DT is not available or fails.
    """
    
    def __init__(self, graph_data, start_node, delivery_nodes, base_speed, 
                 dt_planner=None, use_dt=True):
        """
        Initialize worker
        
        Args:
            graph_data: NetworkX graph
            start_node: Starting hub node
            delivery_nodes: List of delivery nodes
            base_speed: Base speed in m/min
            dt_planner: DecisionTransformerPlanner instance (optional)
            use_dt: Whether to use DT (True) or NNA (False)
        """
        self.G = graph_data
        self.start_node = start_node
        self.delivery_nodes = delivery_nodes
        self.base_speed = base_speed
        self.dt_planner = dt_planner
        self.use_dt = use_dt and dt_planner is not None
        
        if self.use_dt:
            print("✓ Using Decision Transformer for routing")
        else:
            print("⚠ Using Nearest Neighbor Algorithm")
    
    def activate_disasters(self, rain_intensity):
        """Activate disasters based on rain intensity"""
        rain_params = rain_intensity_effects(rain_intensity)
        flood_threshold = rain_params["flood_activation"]
        landslide_threshold = rain_params["landslide_activation"]
        
        if rain_intensity >= 4:
            flood_threshold *= 1.2
            landslide_threshold *= 1.3
        
        activated_disasters = {"floods": 0, "landslides": 0}
        
        for u, v, data in self.G.edges(data=True):
            if random.random() < data['flood_prone'] * flood_threshold:
                data['currently_flooded'] = True
                activated_disasters["floods"] += 1
            else:
                data['currently_flooded'] = False
                
            if random.random() < data['landslide_prone'] * landslide_threshold:
                data['currently_landslide'] = True
                activated_disasters["landslides"] += 1
            else:
                data['currently_landslide'] = False
        
        return activated_disasters
    
    def find_disaster_aware_route(self, start_node, delivery_nodes, rain_intensity):
        """
        Find route using Decision Transformer or NNA fallback
        
        Args:
            start_node: Starting node
            delivery_nodes: Nodes to visit
            rain_intensity: Weather severity
        
        Returns:
            tuple: (route, total_distance, path_sequence)
        """
        if self.use_dt:
            try:
                # Use Decision Transformer
                route, estimated_time = self.dt_planner.plan_route(
                    graph=self.G,
                    start_node=start_node,
                    delivery_nodes=delivery_nodes,
                    rain_intensity=rain_intensity,
                    target_return=-30.0,
                    temperature=0.0  # Greedy
                )
                
                # Convert route to path sequence
                path_sequence = [start_node]
                total_distance = 0
                
                for i in range(len(route) - 1):
                    path, distance = self.find_danger_aware_shortest_path(
                        route[i], route[i+1], rain_intensity
                    )
                    
                    if path:
                        path_sequence.extend(path[1:])
                        total_distance += distance
                    else:
                        # Path blocked, fallback to NNA
                        print(f"  ⚠ DT route blocked, falling back to NNA")
                        return self._nna_fallback(start_node, delivery_nodes, rain_intensity)
                
                return route, total_distance, path_sequence
                
            except Exception as e:
                print(f"  ⚠ DT error: {e}, falling back to NNA")
                return self._nna_fallback(start_node, delivery_nodes, rain_intensity)
        else:
            # Use NNA
            return self._nna_fallback(start_node, delivery_nodes, rain_intensity)
    
    def _nna_fallback(self, start_node, delivery_nodes, rain_intensity):
        """Original Nearest Neighbor Algorithm"""
        unvisited = set(delivery_nodes)
        current_node = start_node
        route = [start_node]
        path_sequence = [start_node]
        total_distance = 0
        
        while unvisited:
            next_node = None
            shortest_distance = float('inf')
            shortest_path = None
            
            for node in unvisited:
                path, distance = self.find_danger_aware_shortest_path(
                    current_node, node, rain_intensity
                )
                if path and distance < shortest_distance:
                    shortest_distance = distance
                    shortest_path = path
                    next_node = node
            
            if next_node:
                route.append(next_node)
                path_sequence.extend(shortest_path[1:])
                total_distance += shortest_distance
                unvisited.remove(next_node)
                current_node = next_node
            else:
                return None, 0, None
        
        return route, total_distance, path_sequence
    
    def find_danger_aware_shortest_path(self, source_id, target_id, rain_intensity, weight='length'):
        """Find shortest path with disaster awareness"""
        G_temp = self.G.copy()
        rain_params = rain_intensity_effects(rain_intensity)
        
        for u, v, data in G_temp.edges(data=True):
            edge_weight = data.get(weight, 1.0)
            penalty_factor = 1.0
            
            if data.get('currently_flooded', False):
                if rain_intensity >= 5:
                    penalty_factor = float('inf')
                else:
                    penalty_factor *= 5.0
            
            if data.get('currently_landslide', False):
                if rain_intensity >= 4:
                    penalty_factor = float('inf')
                else:
                    penalty_factor *= 10.0
            
            penalty_factor /= max(0.1, rain_params["speed_multiplier"])
            adjusted_weight = edge_weight * penalty_factor
            G_temp.edges[u, v]['adjusted_weight'] = adjusted_weight
        
        if source_id not in G_temp.nodes or target_id not in G_temp.nodes:
            return None, float('inf')
        
        try:
            path = nx.shortest_path(G_temp, source=source_id, target=target_id, 
                                   weight='adjusted_weight')
            path_length = sum(
                G_temp.edges[path[i], path[i+1]].get(weight, 1)
                for i in range(len(path)-1)
                if G_temp.has_edge(path[i], path[i+1])
            )
            return path, path_length
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None, float('inf')