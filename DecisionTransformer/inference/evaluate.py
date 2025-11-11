"""
Evaluation and comparison functions
"""
from .planner import DecisionTransformerPlanner
from DecisionTransformer.Utilities import NetworkMap


def example_usage():
    """
    Example of how to use the Decision Transformer planner
    """
    print("="*60)
    print("Decision Transformer Inference Example")
    print("="*60)
    
    # Load planner
    planner = DecisionTransformerPlanner(
        model_path='checkpoints/best_model.pt',
        encoders_path='data/processed/encoders.pkl'
    )
    
    # Load network
    print("\nLoading network...")
    place_query = "La Trinidad, Benguet, Philippines"
    network_map = NetworkMap(place_query)
    network_map.download_map()
    network_map.network_to_networkx()
    
    start_node, delivery_nodes = network_map.select_fixed_points(num_delivery_points=20)
    network_map.initialize_edge_disaster_attributes()
    
    # Plan routes for different weather conditions
    for weather_severity in [1, 3, 5]:
        print(f"\n{'='*60}")
        print(f"Weather Severity: {weather_severity}")
        print(f"{'='*60}")
        
        route, estimated_time = planner.plan_route(
            graph=network_map.G,
            start_node=start_node,
            delivery_nodes=delivery_nodes,
            rain_intensity=weather_severity,
            target_return=-30.0
        )
        
        print(f"Route: {route[:5]}... (showing first 5 nodes)")
        print(f"Estimated time: {estimated_time:.2f} minutes")
        print(f"Number of stops: {len(route) - 1}")


def compare_dt_vs_nna(model_path='checkpoints/best_model.pt', 
                      encoders_path='data/processed/encoders.pkl',
                      num_trials=10):
    """
    Compare Decision Transformer against NNA baseline
    
    Args:
        model_path: Path to trained DT model
        encoders_path: Path to encoders
        num_trials: Number of comparison trials
    """
    import numpy as np
    
    print("="*60)
    print("Comparing Decision Transformer vs Nearest Neighbor Algorithm")
    print("="*60)
    
    # Load DT planner
    dt_planner = DecisionTransformerPlanner(model_path, encoders_path)
    
    # Load network
    print("\nLoading network...")
    place_query = "La Trinidad, Benguet, Philippines"
    network_map = NetworkMap(place_query)
    network_map.download_map()
    network_map.network_to_networkx()
    start_node, delivery_nodes = network_map.select_fixed_points(num_delivery_points=20)
    
    # Import worker classes
    from DecisionTransformer.integration.simulation_worker_dt import MonteCarloSimulationWorkerWithDT
    
    # Create workers
    dt_worker = MonteCarloSimulationWorkerWithDT(
        network_map.G, start_node, delivery_nodes, 500,
        dt_planner=dt_planner, use_dt=True
    )
    
    nna_worker = MonteCarloSimulationWorkerWithDT(
        network_map.G, start_node, delivery_nodes, 500,
        dt_planner=None, use_dt=False
    )
    
    # Run comparison
    results = {'DT': [], 'NNA': []}
    
    for trial in range(num_trials):
        print(f"\nTrial {trial + 1}/{num_trials}")
        
        # Initialize disasters
        network_map.initialize_edge_disaster_attributes()
        
        for rain_intensity in [1, 3, 5]:
            # Activate disasters
            dt_worker.activate_disasters(rain_intensity)
            
            # DT route
            dt_route, dt_dist, dt_path = dt_worker.find_disaster_aware_route(
                start_node, delivery_nodes, rain_intensity
            )
            
            # NNA route
            nna_route, nna_dist, nna_path = nna_worker.find_disaster_aware_route(
                start_node, delivery_nodes, rain_intensity
            )
            
            # Compare
            if dt_route and nna_route:
                improvement = (nna_dist - dt_dist) / nna_dist * 100
                print(f"  Rain {rain_intensity}: DT vs NNA distance: "
                      f"{dt_dist:.0f}m vs {nna_dist:.0f}m "
                      f"({improvement:+.1f}%)")
                
                results['DT'].append(dt_dist)
                results['NNA'].append(nna_dist)
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"DT avg distance: {np.mean(results['DT']):.0f}m")
    print(f"NNA avg distance: {np.mean(results['NNA']):.0f}m")
    improvement = (np.mean(results['NNA']) - np.mean(results['DT'])) / np.mean(results['NNA']) * 100
    print(f"Average improvement: {improvement:.1f}%")

