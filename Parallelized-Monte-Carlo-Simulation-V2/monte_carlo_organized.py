import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
import time
import networkx as nx

from multiprocessing import Process, Queue, current_process, Manager

from Utilities import NetworkMap

from Simulation import MonteCarloSimulation

from Utilities import visualize_combined_simulation_results, plot_deliveries_per_rain_intensity_combined




def process_worker(G_serialized,start_node,delivery_nodes, ideal_delivery_time, weather_severity, num_simulations, main_queue):
    """
    Worker function that creates and runs a full Monte Carlo simulation
    """

    worker_name = current_process().name
    print(f"{worker_name} starting work with weather severity {weather_severity}")

    # Deserialize graph data
    G = nx.node_link_graph(G_serialized, edges="links")

    try:
        # Run full simulation on weather condition
        simulation = MonteCarloSimulation(
                        G, 
                        start_node, 
                        delivery_nodes, 
                        ideal_delivery_time,
                        worker_name)
        
        simulation.run_simulation(num_simulations=num_simulations,weather_severity=weather_severity)
    except Exception as e:
        main_queue.put(None)
        print(f"An error encountered during the simulation with weather_severity {weather_severity}. {e}")

    # Enqueue the result
    main_queue.put(simulation)

    print(f"{worker_name} completed {num_simulations} simulations with weather severity {weather_severity}")





if __name__ == '__main__':
   
    try:
        
        place_query = "La Trinidad, Benguet, Philippines"


        """
        ============================================================================================================================================
        🛠️ Initialize and Download Map
        ============================================================================================================================================
        """
        start = time.time()

        # Initialize NetworkMap class to process place query
        Network_Map_Instance = NetworkMap(place_query)

        # Download place query using osmnx
        Network_Map_Instance.download_map()
        
        # Convert to NetworkX for analysis
        Network_Map_Instance.network_to_networkx()

        start_node, delivery_nodes = Network_Map_Instance.select_fixed_points(num_delivery_points=20)
        print(f"Start node is: {start_node}")
        print(f"Delivery nodes are: {delivery_nodes}")

        # Find route using nearest neighbor
        route, total_distance, path_sequence = Network_Map_Instance.nearest_neighbor_route()

        # Calculate ideal travel time
        Network_Map_Instance.calculate_travel_time(total_distance)


        print(f"Total distance: {total_distance:.2f} meters")
        print(f"Total travel time: {Network_Map_Instance.ideal_delivery_time:.2f} minutes")

        # Initialize random disaster prone nodes
        landslide_prone, flood_prone = Network_Map_Instance.initialize_edge_disaster_attributes()
        

        print(f"Total number of edges: {len(list(Network_Map_Instance.G.edges))}")
        print(f"Number of Landslide prone edges: {len(landslide_prone)}, Number of flood prone edges: {len(flood_prone)}")
        
        
    
        # Visualize the delivery route
        # fig, ax = Network_Map_Instance.visualize_delivery_nodes()
        # plt.show()

        # fig, ax = Network_Map_Instance.visualize_delivery_route()
        # plt.show()

        # fig, ax = Network_Map_Instance.visualize_delivery_route_with_disaster(landslide_prone, flood_prone)
        # plt.show()



        """
        =============================================================================================================================================
        🚀 Run Simulations
        =============================================================================================================================================
        """

        simulation_runs = []
        processes = []
        # start = time.time()

        NUM_SIMULATIONS = 100

        # Serialize networkx graph
        graph_data = nx.node_link_data(Network_Map_Instance.G, edges="links")


        # Initialize queue for processes to share the simulation results
        main_queue = Queue()
        
        # Start all processes
        for weather_severity in range(1, 6):
            p = Process(
                target=process_worker,
                args=(
                    graph_data,
                    Network_Map_Instance.start_node, 
                    Network_Map_Instance.delivery_nodes, 
                    Network_Map_Instance.ideal_delivery_time,
                    weather_severity,
                    NUM_SIMULATIONS,
                    main_queue),
                name=f"Process-{weather_severity}"
            )
            processes.append(p)
            p.start()
        
        # Collect results from the queue
        for _ in range(len(processes)):
            try:
                # Blocked until a worker puts results in the queue
                process_results = main_queue.get()
                # Check if we got a valid result
                if process_results is not None: 
                    simulation_runs.append(process_results)
                    print(f"Received results for weather severity {process_results.weather_severity}")
            except Exception as e:
                print(f"Error receiving results: {e}")
        
        # Wait for all processes to finish
        for p in processes:
            p.join(timeout=10)
            
        end = time.time()
        elapsed_time = end - start
        print(f"Total elapsed time: {elapsed_time:.4f} seconds")

        if simulation_runs:
            print("Starting visualization...")
            # Visualize results
            plot_deliveries_per_rain_intensity_combined(simulation_runs)
            visualize_combined_simulation_results(simulation_runs)
            print("Visualization complete.")
        else:
            print("Error: No simulation results to visualize!")

                



    except Exception as e:
        print(f"Error during main function execution: {e}")



