
from math import ceil

import networkx as nx
import numpy as np

import time
from .SimulationWorker import MonteCarloSimulationWorker
import os





class MonteCarloSimulation:
    
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



    def run_simulation(self, num_simulations=100, weather_severity=3):
        """
        Run multiple delivery simulations with varying rain conditions and improved progress reporting
        """
        print("\n=== STARTING MONTE CARLO SIMULATION ===")
        print(f"Weather condition: {weather_severity} (1=Light, 5=Typhoon)")
        
        start_time = time.time()

        self.weather_severity = weather_severity
        simulator = MonteCarloSimulationWorker(self.G, self.start_node, self.delivery_nodes, self.base_speed)
        

        for i in range(num_simulations):
            rain_intensity = np.random.choice(range(1,6), p=simulator.get_rainfall_probability_by_condition(weather_severity))
            activated_disasters = simulator.activate_disasters(rain_intensity)
            result = simulator.simulate_delivery(rain_intensity)

            result['rain_intensity'] = rain_intensity
            result['activated_disasters'] = activated_disasters
            self.simulation_results['all_runs'].append(result)

            self.completed_simulations = i + 1

            progress = (self.completed_simulations/num_simulations)*100
            if (i + 1) % 1 == 0:  # Print every simulation
                print(f"{self.worker_name} Progress: {self.completed_simulations}/{num_simulations} "
                      f"({progress:.1f}%) - Current rain intensity: {rain_intensity} "
                      f"- Floods: {activated_disasters['floods']} "
                      f"- Landslides: {activated_disasters['landslides']}")

        elapsed_time = time.time() - start_time
        print(f"\n=== SIMULATION COMPLETE ===")
        print(f"Finished {self.completed_simulations} simulations in {elapsed_time:.2f} seconds")
        
        # Calculate success rate
        success_count = sum(1 for r in self.simulation_results['all_runs'] if r.get('success', False))
        print(f"Success rate: {success_count}/{len(self.simulation_results['all_runs'])} ({success_count/len(self.simulation_results['all_runs'])*100:.1f}%)")
        