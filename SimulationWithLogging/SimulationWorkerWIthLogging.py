"""
Enhanced SimulationWorker that logs detailed trajectories for Decision Transformer training
"""
from typing import Dict, Any, Tuple, List
import networkx as nx
import random
import numpy as np


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


class MonteCarloSimulationWorkerWithLogging:
    """
    Extended version that captures step-by-step trajectory data
    """
    def __init__(self, graph_data, start_node, delivery_nodes, base_speed):
        self.G = graph_data
        self.start_node = start_node
        self.delivery_nodes = delivery_nodes
        self.base_speed = base_speed
    
    def get_rainfall_probability_by_condition(self, weather_severity):
        return {
            1: [0.92, 0.06, 0.015, 0.004, 0.001],
            2: [0.55, 0.25, 0.12, 0.06, 0.02],
            3: [0.25, 0.25, 0.25, 0.15, 0.10],
            4: [0.05, 0.10, 0.20, 0.30, 0.35],
            5: [0.005, 0.02, 0.075, 0.35, 0.55]
        }[weather_severity]
    
    def activate_disasters(self, rain_intensity):
        """Activate disasters based on rain level"""
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
    
    def get_disaster_context(self, node, radius=3):
        """
        Get disaster information within radius hops of a node
        
        Returns:
            dict with flood and landslide counts nearby
        """
        # Get nodes within radius
        try:
            nearby_nodes = nx.single_source_shortest_path_length(
                self.G, node, cutoff=radius
            ).keys()
        except:
            nearby_nodes = [node]
        
        # Count disasters on edges connected to nearby nodes
        floods_nearby = 0
        landslides_nearby = 0
        
        for u in nearby_nodes:
            for v in self.G.neighbors(u):
                if self.G.has_edge(u, v):
                    edge_data = self.G.edges[u, v]
                    if edge_data.get('currently_flooded', False):
                        floods_nearby += 1
                    if edge_data.get('currently_landslide', False):
                        landslides_nearby += 1
        
        return {
            'floods_nearby': floods_nearby,
            'landslides_nearby': landslides_nearby
        }
    
    def simulate_delivery_with_trajectory(self, rain_intensity):
        """
        NEW: Simulate delivery and capture detailed trajectory
        
        Returns:
            dict with:
                - trajectory: list of step-by-step data
                - total_return: sum of rewards
                - success: whether all deliveries completed
                - aggregate_stats: original simulation statistics
        """
        rain_params = rain_intensity_effects(rain_intensity)
        
        # Find initial route
        route, total_distance, path_sequence = self.find_disaster_aware_route(
            self.start_node, self.delivery_nodes, rain_intensity
        )
        
        if not route:
            return {
                'trajectory': [],
                'total_return': -1000.0,  # Large penalty
                'success': False,
                'aggregate_stats': {
                    'success': False,
                    'distance': 0,
                    'time': 0,
                    'reroutes': 0,
                    'reason': 'No valid route found initially'
                }
            }
        
        # Simulate movement and LOG TRAJECTORY
        # trajectory, success, total_time, total_distance, deliveries_made, disaster_encounters, reason = \
        #     self.simulate_movement_with_logging(route, path_sequence, rain_intensity)
        
        # # Compute total return (negative time for minimization)
        # total_return = -total_time
        
        # return {
        #     'trajectory': trajectory,
        #     'total_return': total_return,
        #     'success': success,
        #     'rain_intensity': rain_intensity,
        #     'aggregate_stats': {
        #         'success': success,
        #         'distance': total_distance,
        #         'time': total_time,
        #         'deliveries_made': deliveries_made,
        #         'disaster_encounters': disaster_encounters,
        #         'reason': reason,
        #         'reroutes': 0
        #     }

        trajectory, success, total_time, total_distance, deliveries_made, disaster_encounters, reason = \
        self.simulate_movement_with_logging(route, path_sequence, rain_intensity)

        total_return = -total_time

        # Flatten states, actions, rewards
        states = [step['state'] for step in trajectory]
        actions = [step['action'] for step in trajectory]
        rewards = [step['reward'] for step in trajectory]

        return {
            'trajectory': trajectory,
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'total_reward': total_return,
            'success': success,
            'rain_intensity': rain_intensity,
            'aggregate_stats': {
                'success': success,
                'distance': total_distance,
                'time': total_time,
                'deliveries_made': deliveries_made,
                'disaster_encounters': disaster_encounters,
                'reason': reason,
                'reroutes': 0
            }
        }
    
    def simulate_movement_with_logging(self, route, path_sequence, rain_intensity):
        """
        NEW: Simulate movement and log each decision step
        
        Returns:
            trajectory: List of step dictionaries
            success: Boolean
            total_time: Float
            total_distance: Float
            deliveries_made: Int
            disaster_encounters: Dict
            reason: String
        """
        trajectory = []
        current_node = self.start_node
        current_path_idx = 0
        total_distance = 0
        total_time = 0
        disaster_encounters = {"floods": 0, "landslides": 0}
        visited_delivery_nodes = set()
        remaining_delivery_nodes = set(self.delivery_nodes)
        delivery_nodes_set = set(route[1:])
        
        # Track position in route for each delivery decision
        delivery_step = 0
        
        for i in range(len(route) - 1):
            # Current and next delivery nodes
            from_delivery_node = route[i]
            to_delivery_node = route[i + 1]
            
            # CAPTURE STATE BEFORE MAKING DECISION
            state = {
                'current_node': from_delivery_node,
                'remaining_nodes': sorted(list(remaining_delivery_nodes)),
                'rain_intensity': rain_intensity,
                'position_in_route': delivery_step,
                'total_nodes': len(self.delivery_nodes),
                'disaster_context': self.get_disaster_context(from_delivery_node)
            }
            
            # ACTION: Next delivery node chosen
            action = to_delivery_node
            
            # Find path between delivery nodes
            segment_path, segment_distance = self.find_danger_aware_shortest_path(
                from_delivery_node, to_delivery_node, rain_intensity
            )
            
            if not segment_path:
                # Path blocked - trajectory ends here
                trajectory.append({
                    'state': state,
                    'action': action,
                    'reward': -100.0,  # Large penalty for blockage
                    'next_state': state,  # Same state (stuck)
                    'step_time': 100.0,
                    'step_distance': 0,
                    'blocked': True
                })
                return (trajectory, False, total_time, total_distance, 
                        len(visited_delivery_nodes), disaster_encounters, 
                        "Encountered blocked road")
            
            # Simulate traversal along segment_path
            segment_time = 0
            segment_disasters = {"floods": 0, "landslides": 0}
            
            for j in range(len(segment_path) - 1):
                u = segment_path[j]
                v = segment_path[j + 1]
                
                edge_data = self.G.edges[u, v]
                
                # Check if blocked
                if self.is_edge_blocked(edge_data, rain_intensity):
                    trajectory.append({
                        'state': state,
                        'action': action,
                        'reward': -100.0,
                        'next_state': state,
                        'step_time': 100.0,
                        'step_distance': 0,
                        'blocked': True
                    })
                    return (trajectory, False, total_time, total_distance,
                            len(visited_delivery_nodes), disaster_encounters,
                            "Encountered blocked road")
                
                # Count disasters
                if edge_data.get('currently_flooded', False):
                    segment_disasters["floods"] += 1
                    disaster_encounters["floods"] += 1
                if edge_data.get('currently_landslide', False):
                    segment_disasters["landslides"] += 1
                    disaster_encounters["landslides"] += 1
                
                # Calculate travel time
                adjusted_speed = self.base_speed * rain_intensity_effects(rain_intensity)["speed_multiplier"]
                if edge_data.get('currently_flooded', False):
                    adjusted_speed *= 0.5
                if edge_data.get('currently_landslide', False):
                    adjusted_speed *= 0.3
                
                if adjusted_speed <= 0:
                    trajectory.append({
                        'state': state,
                        'action': action,
                        'reward': -100.0,
                        'next_state': state,
                        'step_time': 100.0,
                        'step_distance': 0,
                        'blocked': True
                    })
                    return (trajectory, False, total_time, total_distance,
                            len(visited_delivery_nodes), disaster_encounters,
                            "Speed dropped to zero")
                
                edge_length = edge_data.get('length', 0)
                travel_time = edge_length / adjusted_speed
                
                segment_time += travel_time
            
            # Update totals
            total_time += segment_time
            total_distance += segment_distance
            
            # REWARD: Negative time (we want to minimize time)
            reward = -segment_time
            
            # Update remaining nodes
            remaining_delivery_nodes.discard(to_delivery_node)
            visited_delivery_nodes.add(to_delivery_node)
            
            # CAPTURE NEXT STATE
            next_state = {
                'current_node': to_delivery_node,
                'remaining_nodes': sorted(list(remaining_delivery_nodes)),
                'rain_intensity': rain_intensity,
                'position_in_route': delivery_step + 1,
                'total_nodes': len(self.delivery_nodes),
                'disaster_context': self.get_disaster_context(to_delivery_node)
            }
            
            # LOG THIS STEP
            trajectory.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'step_time': segment_time,
                'step_distance': segment_distance,
                'disasters_encountered': segment_disasters.copy(),
                'blocked': False
            })
            
            delivery_step += 1
            current_node = to_delivery_node
        
        # Check success
        success = len(remaining_delivery_nodes) == 0
        reason = "All deliveries completed" if success else "Some deliveries missed"
        
        return (trajectory, success, total_time, total_distance,
                len(visited_delivery_nodes), disaster_encounters, reason)
    
    # Keep existing helper methods
    def is_edge_blocked(self, edge_data, rain_intensity):
        if edge_data.get('currently_flooded', False) and rain_intensity >= 5:
            return True
        if edge_data.get('currently_landslide', False) and rain_intensity >= 4:
            return True
        return False
    
    def find_disaster_aware_route(self, start_node, delivery_nodes, rain_intensity):
        """Find route using nearest neighbor heuristic"""
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