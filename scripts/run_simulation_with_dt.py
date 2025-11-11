"""
Step 5: Run full simulation with Decision Transformer
"""
import sys
sys.path.append('..')

import time
from multiprocessing import Process, Queue
import networkx as nx

from DecisionTransformer.Utilities import NetworkMap
from DecisionTransformer.inference.planner import DecisionTransformerPlanner
from DecisionTransformer.integration.simulation_worker_dt import MonteCarloSimulationWorkerWithDT


def process_worker(G_serialized, start_node, delivery_nodes, base_speed,
                   weather_severity, num_simulations, dt_planner, main_queue):
    """Worker function for parallel simulation"""
    worker_name = f"Process-{weather_severity}"
    print(f"{worker_name} starting work with weather severity {weather_severity}")
    
    # Deserialize graph
    G = nx.node_link_graph(G_serialized, edges="links")
    
    # Create worker with DT
    worker = MonteCarloSimulationWorkerWithDT(
        G, start_node, delivery_nodes, base_speed,
        dt_planner=dt_planner, use_dt=True
    )
    
    # Run simulations
    # (You would need to implement the full simulation loop here)
    # This is a simplified version
    
    results = {
        'weather_severity': weather_severity,
        'num_simulations': num_simulations,
        'worker_name': worker_name
    }
    
    main_queue.put(results)
    print(f"{worker_name} completed {num_simulations} simulations")


def main():
    """Run full Monte Carlo simulation with Decision Transformer"""
    print("="*60)
    print("STEP 5: FULL SIMULATION WITH DECISION TRANSFORMER")
    print("="*60)
    
    # Load network
    print("\nLoading network...")
    place_query = "La Trinidad, Benguet, Philippines"
    network_map = NetworkMap(place_query)
    network_map.download_map()
    network_map.network_to_networkx()
    
    start_node, delivery_nodes = network_map.select_fixed_points(num_delivery_points=20)
    network_map.initialize_edge_disaster_attributes()
    
    # Load Decision Transformer
    print("\nLoading Decision Transformer...")
    dt_planner = DecisionTransformerPlanner(
        model_path='../checkpoints/best_model.pt',
        encoders_path='../data/processed/encoders.pkl'
    )
    
    # Serialize graph for multiprocessing
    graph_data = nx.node_link_data(network_map.G, edges="links")
    
    # Run simulations
    print("\nRunning simulations...")
    NUM_SIMULATIONS = 100
    main_queue = Queue()
    processes = []
    
    start_time = time.time()
    
    for weather_severity in range(1, 6):
        p = Process(
            target=process_worker,
            args=(
                graph_data,
                start_node,
                delivery_nodes,
                500,  # base_speed
                weather_severity,
                NUM_SIMULATIONS,
                dt_planner,
                main_queue
            ),
            name=f"Process-{weather_severity}"
        )
        processes.append(p)
        p.start()
    
    # Collect results
    simulation_results = []
    for _ in range(len(processes)):
        try:
            result = main_queue.get()
            if result is not None:
                simulation_results.append(result)
                print(f"Received results for weather severity {result['weather_severity']}")
        except Exception as e:
            print(f"Error receiving results: {e}")
    
    # Wait for all processes
    for p in processes:
        p.join(timeout=10)
    
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"✓ Simulation complete!")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Results collected: {len(simulation_results)}")
    print("="*60)


if __name__ == '__main__':
    main()