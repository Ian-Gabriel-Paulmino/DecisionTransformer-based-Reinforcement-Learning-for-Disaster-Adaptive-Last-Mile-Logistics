
from math import ceil

import networkx as nx
import numpy as np
import pickle

import time
from .SimulationWorkerWIthLogging import MonteCarloSimulationWorkerWithLogging
import os





class MonteCarloSimulationWithLogging:
    
    def __init__(self, G, start_node, delivery_nodes, ideal_delivery_time, worker_name ,base_speed_mpm=500, num_delivery_points=7):
        """
        Initialize the delivery simulation with disaster awareness
        
        Args:
            G: networkx graph
            start_node: simulates delivery hub / sorting center
            delivery_nodes: simulates delivery locations
            base_speed_mpm: Base speed in meters per minute (default: 500 m/min = 30 km/h)
            num_delivery_points: Number of delivery points to select
        """
        self.base_speed = base_speed_mpm
        self.num_delivery_points = num_delivery_points
        self.G = G
        
        self.completed_simulations = 0
        
        # Select start and delivery nodes
        self.start_node = start_node
        self.delivery_nodes = delivery_nodes

        self.ideal_delivery_time = ideal_delivery_time
        self.weather_severity = None

        # Stats tracking
        self.simulation_results = {
            'all_runs': [],
        }

        # Keeps track of the name of the process that is working with this siumalation instance
        self.worker_name = worker_name



    def run_simulation_with_logging(self, num_simulations=100, weather_severity=3, 
                                        save_path='trajectories.pkl'):
        """
        Run simulations and save trajectory data for Decision Transformer training
        """
        print(f"\n=== COLLECTING TRAJECTORY DATA ===")
        print(f"Weather severity: {weather_severity}")
        
        self.weather_severity = weather_severity
        simulator = MonteCarloSimulationWorkerWithLogging(
            self.G, self.start_node, self.delivery_nodes, self.base_speed
        )
        
        all_trajectories = []
        
        for i in range(num_simulations):
            rain_intensity = np.random.choice(
                range(1, 6), 
                p=simulator.get_rainfall_probability_by_condition(weather_severity)
            )
            
            activated_disasters = simulator.activate_disasters(rain_intensity)
            
            # NEW: Get trajectory data
            result = simulator.simulate_delivery_with_trajectory(rain_intensity)
            result['activated_disasters'] = activated_disasters
            result['weather_severity'] = weather_severity
            result['simulation_id'] = i
            
            all_trajectories.append(result)
            
            if (i + 1) % 10 == 0:
                print(f"Progress: {i+1}/{num_simulations} - "
                    f"Last trajectory: {len(result['trajectory'])} steps, "
                    f"Success: {result['success']}")
        
        # Save trajectories
        with open(save_path, 'wb') as f:
            pickle.dump(all_trajectories, f)
        
        print(f"\n✓ Saved {len(all_trajectories)} trajectories to {save_path}")
        print(f"Total steps collected: {sum(len(t['trajectory']) for t in all_trajectories)}")
        
        return all_trajectories