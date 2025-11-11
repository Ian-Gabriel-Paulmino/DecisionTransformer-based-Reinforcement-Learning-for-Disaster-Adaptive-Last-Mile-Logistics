from typing import Dict, Any, Tuple

import networkx as nx
import random



def rain_intensity_effects(rain_intensity):
    """Get disaster activation parameters based on rain level"""
    rain_effects = {
        1: {"flood_activation": 0.10, "landslide_activation": 0.00, "speed_multiplier": 1.0},  # Light rain
        2: {"flood_activation": 0.30, "landslide_activation": 0.05, "speed_multiplier": 0.8},  # Moderate rain
        3: {"flood_activation": 0.60, "landslide_activation": 0.15, "speed_multiplier": 0.6},  # Heavy rain
        4: {"flood_activation": 0.90, "landslide_activation": 0.30, "speed_multiplier": 0.4},  # Typhoon
        5: {"flood_activation": 1.00, "landslide_activation": 1.00, "speed_multiplier": 0.0}   # Extreme typhoon
    }
    return rain_effects.get(rain_intensity, rain_effects[1])  # Default to level 1 if invalid




class MonteCarloSimulationWorker:
    def __init__(self, graph_data, start_node, delivery_nodes, base_speed):
        
        # Rebuild graph from serialized form
        # self.G = nx.node_link_graph(graph_data, edges="links")
        self.G = graph_data

        self.start_node = start_node
        self.delivery_nodes = delivery_nodes
        self.base_speed = base_speed

    def get_rainfall_probability_by_condition(self, weather_severity):
        return {
            1: [0.92, 0.06, 0.015, 0.004, 0.001],  # Mostly light rain
            2: [0.55, 0.25, 0.12, 0.06, 0.02],     # Some moderate, occasional heavy
            3: [0.25, 0.25, 0.25, 0.15, 0.10],     # Uniform midrange, slight tilt toward extremes
            4: [0.05, 0.10, 0.20, 0.30, 0.35],     # Shifted toward extreme typhoons
            5: [0.005, 0.02, 0.075, 0.35, 0.55]    # Dominated by extreme effects
        }[weather_severity]
    

    def activate_disasters(self, rain_intensity):
        """
        Activate disasters based on rain level and edge disaster probabilities
        
        Args:
            rain_intensity: Integer from 1-5 representing rain intensity
        """
        rain_params = rain_intensity_effects(rain_intensity)

        # Extract ativations values from rain parameters
        flood_threshold = rain_params["flood_activation"]
        landslide_threshold = rain_params["landslide_activation"]

        # Increases the threshold during Stormy or Typhoon conditions
        if rain_intensity >= 4:
            flood_threshold *= 1.2
            landslide_threshold *= 1.3
        
        activated_disasters = {"floods": 0, "landslides": 0}
        
        for u, v, data in self.G.edges(data=True):
            # Check for flood activation
            if random.random() < data['flood_prone'] * flood_threshold:
                data['currently_flooded'] = True
                activated_disasters["floods"] += 1
            else:
                data['currently_flooded'] = False
                
            # Check for landslide activation
            if random.random() < data['landslide_prone'] * landslide_threshold:
                data['currently_landslide'] = True
                activated_disasters["landslides"] += 1
            else:
                data['currently_landslide'] = False
        
        print(f"Activated {activated_disasters['floods']} floods and {activated_disasters['landslides']} landslides")
        return activated_disasters
    

    
    def simulate_delivery(self, rain_intensity):
        """
        Simulate a single delivery attempt with current disaster conditions
        
        Args:
            rain_intensity: Integer from 1-5 representing rain intensity
        
        Returns:
            Dictionary with simulation results
        """
        rain_params = rain_intensity_effects(rain_intensity)
        speed_multiplier = rain_params["speed_multiplier"]
        
        # Find initial route using disaster-aware path finding
        route, total_distance, path_sequence = self.find_disaster_aware_route(
            self.start_node, self.delivery_nodes, rain_intensity)
        
        # If no valid route found
        if not route:
            return {
                'success': False,
                'distance': 0,
                'time': 0,
                'reroutes': 0,
                'disaster_encounters': {},
                'reason': 'No valid route found initially'
            }
        
        # Proceed with movement simulation when there is a valid route
        success, distance, time, deliveries_made, disaster_encounters, reason = self.simulate_movement(
        route, path_sequence, rain_intensity
        )

        return {
            'reroutes': 0,
            'success': success,
            'distance': distance,
            'time': time,
            'deliveries_made': deliveries_made,
            'disaster_encounters': disaster_encounters,
            'reason': reason
        }
    



    def simulate_movement(self, route, path_sequence, rain_intensity):
        current_node = self.start_node
        current_path_idx = 0
        total_distance = 0
        total_time = 0
        disaster_encounters = {"floods": 0, "landslides": 0}
        visited_delivery_nodes = set()

        remaining_delivery_nodes = set(self.delivery_nodes)

        # For easier lookup
        delivery_nodes_set = set(route[1:])  # Start node is not a delivery node

        while current_path_idx < len(path_sequence) - 1:
            u = path_sequence[current_path_idx]
            v = path_sequence[current_path_idx + 1]

            if not self.G.has_edge(u, v):
                # Edge no longer exists (could happen if disaster dynamically deletes edges)
                return False, total_distance, total_time, len(visited_delivery_nodes), disaster_encounters, "Edge no longer exists"

            edge_data = self.G.edges[u, v]

            # Check if edge is blocked
            if self.is_edge_blocked(edge_data, rain_intensity):
                return False, total_distance, total_time, len(visited_delivery_nodes), disaster_encounters, "Encountered blocked road"

            # Count disasters
            if edge_data.get('currently_flooded', False):
                disaster_encounters["floods"] += 1
            if edge_data.get('currently_landslide', False):
                disaster_encounters["landslides"] += 1

            # Adjust speed
            adjusted_speed = self.base_speed * rain_intensity_effects(rain_intensity)["speed_multiplier"]
            if edge_data.get('currently_flooded', False):
                adjusted_speed *= 0.5
            if edge_data.get('currently_landslide', False):
                adjusted_speed *= 0.3

            if adjusted_speed <= 0:
                return False, total_distance, total_time, len(visited_delivery_nodes), disaster_encounters, "Speed dropped to zero"

            # Calculate distance and time
            edge_length = edge_data.get('length', 0)
            travel_time = edge_length / adjusted_speed

            total_distance += edge_length
            total_time += travel_time

            # Move to next node
            current_node = v
            current_path_idx += 1

            # Check if delivery made
            if current_node in delivery_nodes_set:
                visited_delivery_nodes.add(current_node)
                remaining_delivery_nodes.discard(current_node)

        # If we reach here, and no blockages happened
        success = len(remaining_delivery_nodes) == 0
        reason = "All deliveries completed" if success else "Some deliveries missed"

        return success, total_distance, total_time, len(visited_delivery_nodes), disaster_encounters, reason

    def is_edge_blocked(self, edge_data, rain_intensity):
        """
        Determines if an edge is completely blocked based on disaster conditions.
        """
        if edge_data.get('currently_flooded', False) and rain_intensity >= 5:
            return True
        if edge_data.get('currently_landslide', False) and rain_intensity >= 4:
            return True
        return False



    def find_disaster_aware_route(self, start_node, delivery_nodes, rain_intensity):
            """Find route using nearest neighbor heuristic with disaster awareness"""
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
                    path, distance = self.find_danger_aware_shortest_path(current_node, node, rain_intensity)
                    if path and distance < shortest_distance:
                        shortest_distance = distance
                        shortest_path = path
                        next_node = node
                
                if next_node:
                    # Add next node to route
                    route.append(next_node)
                    
                    # Add path to complete sequence (excluding first node since it's already there)
                    path_sequence.extend(shortest_path[1:])
                    total_distance += shortest_distance
                    unvisited.remove(next_node)
                    current_node = next_node
                else:
                    print("Could not find path to remaining delivery nodes")
                    return None, 0, None
            
            return route, total_distance, path_sequence



    def find_danger_aware_shortest_path(self, source_id, target_id, rain_intensity, weight='length'):
        """
        Find shortest path with disaster awareness
        
        Args:
            source_id: Starting node ID
            target_id: Target node ID
            rain_intensity: Current rain level (1-5)
            weight: Edge attribute to use as base weight
            
        Returns:
            Tuple of (path, adjusted_length)
        """
        # Create a copy of the graph for pathfinding with adjusted weights
        G_temp = self.G.copy()
        
        # Get rain parameters
        rain_params = rain_intensity_effects(rain_intensity)
        
        # Adjust weights based on disaster conditions
        for u, v, data in G_temp.edges(data=True):
            edge_weight = data.get(weight, 1.0)
            penalty_factor = 1.0
            
            # Apply penalties for current disasters
            if data.get('currently_flooded', False):
                # Heavy penalty for flooded roads
                if rain_intensity >= 5:  # Complete blockage in extreme conditions
                    penalty_factor = float('inf')
                else:
                    penalty_factor *= 5.0  # 5x penalty for flooded roads
            
            if data.get('currently_landslide', False):
                # Very heavy penalty for landslides
                if rain_intensity >= 4:  # Complete blockage in typhoon conditions
                    penalty_factor = float('inf')
                else:
                    penalty_factor *= 10.0  # 10x penalty for landslide roads
            
            # Apply rain-based general slowdown
            penalty_factor /= max(0.1, rain_params["speed_multiplier"])  # Avoid division by zero
            
            # Update the edge weight with the combined penalty
            adjusted_weight = edge_weight * penalty_factor
            G_temp.edges[u, v]['adjusted_weight'] = adjusted_weight
        
        # Check if nodes exist
        if source_id not in G_temp.nodes or target_id not in G_temp.nodes:
            return None, float('inf')
        
        # Find shortest path
        try:
            path = nx.shortest_path(G_temp, source=source_id, target=target_id, weight='adjusted_weight')
            
            # Calculate path length using original lengths
            path_length = 0
            for i in range(len(path)-1):
                if G_temp.has_edge(path[i], path[i+1]):
                    edge_data = G_temp.get_edge_data(path[i], path[i+1])
                    if weight in edge_data:
                        path_length += edge_data[weight]
                    else:
                        path_length += 1
                else:
                    # Edge doesn't exist in the graph (completely blocked)
                    return None, float('inf')
                    
            return path, path_length
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None, float('inf')