"""
Evaluation and comparison functions
"""
import sys
import os
import networkx as nx

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from .planner import DecisionTransformerPlanner
from DecisionTransformer.Utilities import NetworkMap
from SimulationWithLogging import MonteCarloSimulationWorkerWithLogging



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
            target_return=-140.0
        )
        
        print(f"Route: {route[:5]}... (showing first 5 nodes)")
        print(f"Estimated time: {estimated_time:.2f} minutes")
        print(f"Number of stops: {len(route) - 1}")


# def compare_dt_vs_nna(model_path='checkpoints/best_model.pt', 
#                       encoders_path='data/processed/encoders.pkl',
#                       num_trials=10):
#     """
#     Compare Decision Transformer against NNA baseline
    
#     Args:
#         model_path: Path to trained DT model
#         encoders_path: Path to encoders
#         num_trials: Number of comparison trials
#     """
#     import numpy as np
    
#     print("="*60)
#     print("Comparing Decision Transformer vs Nearest Neighbor Algorithm")
#     print("="*60)
    
#     # Load DT planner
#     dt_planner = DecisionTransformerPlanner(model_path, encoders_path)
    
#     # Load network
#     print("\nLoading network...")
#     place_query = "La Trinidad, Benguet, Philippines"
#     network_map = NetworkMap(place_query)
#     network_map.download_map()
#     network_map.network_to_networkx()
#     start_node, delivery_nodes = network_map.select_fixed_points(num_delivery_points=20)
    
#     # Import worker classes
#     from DecisionTransformer.integration.simulation_worker_dt import MonteCarloSimulationWorkerWithDT
    
#     # Create workers
#     dt_worker = MonteCarloSimulationWorkerWithDT(
#         network_map.G, start_node, delivery_nodes, 500,
#         dt_planner=dt_planner, use_dt=True
#     )
    
#     nna_worker = MonteCarloSimulationWorkerWithDT(
#         network_map.G, start_node, delivery_nodes, 500,
#         dt_planner=None, use_dt=False
#     )
    
#     # Run comparison
#     results = {'DT': [], 'NNA': []}
    
#     for trial in range(num_trials):
#         print(f"\nTrial {trial + 1}/{num_trials}")
        
#         # Initialize disasters
#         network_map.initialize_edge_disaster_attributes()
        
#         for rain_intensity in [1, 3, 5]:
#             # Activate disasters
#             dt_worker.activate_disasters(rain_intensity)
            
#             # DT route
#             dt_route, dt_dist, dt_path = dt_worker.find_disaster_aware_route(
#                 start_node, delivery_nodes, rain_intensity
#             )
            
#             # NNA route
#             nna_route, nna_dist, nna_path = nna_worker.find_disaster_aware_route(
#                 start_node, delivery_nodes, rain_intensity
#             )
            
#             # Compare
#             if dt_route and nna_route:
#                 improvement = (nna_dist - dt_dist) / nna_dist * 100
#                 print(f"  Rain {rain_intensity}: DT vs NNA distance: "
#                       f"{dt_dist:.0f}m vs {nna_dist:.0f}m "
#                       f"({improvement:+.1f}%)")
                
#                 results['DT'].append(dt_dist)
#                 results['NNA'].append(nna_dist)
    
#     # Summary
#     print(f"\n{'='*60}")
#     print("Summary")
#     print(f"{'='*60}")
#     print(f"DT avg distance: {np.mean(results['DT']):.0f}m")
#     print(f"NNA avg distance: {np.mean(results['NNA']):.0f}m")
#     improvement = (np.mean(results['NNA']) - np.mean(results['DT'])) / np.mean(results['NNA']) * 100
#     print(f"Average improvement: {improvement:.1f}%")

def compare_dt_vs_nna(model_path='checkpoints/best_model.pt', 
                      encoders_path='data/processed/encoders.pkl',
                      num_trials=50,
                      num_deliveries=20):
    """
    Properly compare Decision Transformer against NNA baseline
    WITH ADAPTIVE TARGET RETURNS
    """
    import numpy as np
    from collections import defaultdict
    
    print("="*70)
    print("DECISION TRANSFORMER vs NEAREST NEIGHBOR EVALUATION")
    print("WITH ADAPTIVE RETURN-TO-GO CONDITIONING")
    print("="*70)
    
    # Load DT planner
    print("\n📦 Loading Decision Transformer...")
    dt_planner = DecisionTransformerPlanner(model_path, encoders_path)
    
    # Load network
    print("🗺️  Loading La Trinidad network...")
    place_query = "La Trinidad, Benguet, Philippines"
    network_map = NetworkMap(place_query)
    network_map.download_map()
    network_map.network_to_networkx()
    
    # Results storage
    results = defaultdict(lambda: defaultdict(list))
    
    # ========================================
    # CALIBRATION PHASE: Learn realistic targets
    # ========================================
    print("\n🎯 CALIBRATION: Determining realistic targets per weather...")
    print("-" * 70)
    
    calibration_targets = {}
    
    for weather_severity in [1, 2, 3, 4, 5]:
        calibration_times = []
        
        # Run 20 calibration trials with NNA to understand feasible times
        for _ in range(20):
            start_node, delivery_nodes = network_map.select_fixed_points(
                num_delivery_points=num_deliveries
            )
            network_map.initialize_edge_disaster_attributes()
            
            rain_probs = {
                1: [0.92, 0.06, 0.015, 0.004, 0.001],
                2: [0.55, 0.25, 0.12, 0.06, 0.02],
                3: [0.25, 0.25, 0.25, 0.15, 0.10],
                4: [0.05, 0.10, 0.20, 0.30, 0.35],
                5: [0.005, 0.02, 0.075, 0.35, 0.55]
            }
            rain_intensity = np.random.choice(range(1, 6), p=rain_probs[weather_severity])
            
            worker = MonteCarloSimulationWorkerWithLogging(
                network_map.G, start_node, delivery_nodes, 500
            )
            worker.activate_disasters(rain_intensity)
            
            nna_route, _, _ = worker.find_disaster_aware_route(
                start_node, delivery_nodes, rain_intensity
            )
            
            if nna_route:
                success, time, _ = simulate_route_execution(
                    network_map.G, nna_route, rain_intensity, 500
                )
                if success:
                    calibration_times.append(time)
        
        if calibration_times:
            # Set target as top quartile performance (aspirational but realistic)
            target_time = np.percentile(calibration_times, 25)  # 25th percentile = top 25%
            calibration_targets[weather_severity] = -target_time  # Negative for return
            
            print(f"Weather {weather_severity}: "
                  f"Target = {target_time:.1f} min "
                  f"(median: {np.median(calibration_times):.1f}, "
                  f"range: {min(calibration_times):.1f}-{max(calibration_times):.1f})")
        else:
            # Fallback for impossible conditions
            calibration_targets[weather_severity] = -200.0
            print(f"Weather {weather_severity}: No successful calibration routes (extreme conditions)")
    
    print(f"\n✓ Calibration complete. Targets: {calibration_targets}")
    
    # ========================================
    # MAIN EVALUATION WITH MULTIPLE STRATEGIES
    # ========================================
    print("\n" + "="*70)
    print("MAIN EVALUATION")
    print("="*70)
    
    for trial in range(num_trials):
        print(f"\nTrial {trial + 1}/{num_trials}")
        print("-" * 70)
        
        # Select new delivery points
        start_node, delivery_nodes = network_map.select_fixed_points(
            num_delivery_points=num_deliveries
        )
        network_map.initialize_edge_disaster_attributes()
        
        for weather_severity in [1, 2, 3, 4, 5]:
            rain_probs = {
                1: [0.92, 0.06, 0.015, 0.004, 0.001],
                2: [0.55, 0.25, 0.12, 0.06, 0.02],
                3: [0.25, 0.25, 0.25, 0.15, 0.10],
                4: [0.05, 0.10, 0.20, 0.30, 0.35],
                5: [0.005, 0.02, 0.075, 0.35, 0.55]
            }
            rain_intensity = np.random.choice(range(1, 6), p=rain_probs[weather_severity])
            
            worker = MonteCarloSimulationWorkerWithLogging(
                network_map.G, start_node, delivery_nodes, 500
            )
            activated = worker.activate_disasters(rain_intensity)
            
            # ========================================
            # STRATEGY 1: DT with Aspirational Target
            # ========================================
            target_return = calibration_targets[weather_severity]
            
            try:
                dt_route_aspirational, _ = dt_planner.plan_route(
                    graph=network_map.G,
                    start_node=start_node,
                    delivery_nodes=delivery_nodes,
                    rain_intensity=rain_intensity,
                    target_return=target_return,  # ✅ Weather-appropriate target
                    max_steps=num_deliveries,
                    temperature=0.0
                )
                
                dt_success, dt_time, dt_distance = simulate_route_execution(
                    network_map.G, dt_route_aspirational, rain_intensity, 500
                )
                dt_deliveries = len(set(dt_route_aspirational) & set(delivery_nodes))
                
            except Exception as e:
                dt_success = False
                dt_time = 999999
                dt_distance = 999999
                dt_deliveries = 0
            
            # ========================================
            # STRATEGY 2: DT with Conservative Target
            # ========================================
            conservative_target = calibration_targets[weather_severity] * 1.5  # 50% more time
            
            try:
                dt_route_conservative, _ = dt_planner.plan_route(
                    graph=network_map.G,
                    start_node=start_node,
                    delivery_nodes=delivery_nodes,
                    rain_intensity=rain_intensity,
                    target_return=conservative_target,
                    max_steps=num_deliveries,
                    temperature=0.0
                )
                
                dt_cons_success, dt_cons_time, dt_cons_distance = simulate_route_execution(
                    network_map.G, dt_route_conservative, rain_intensity, 500
                )
                dt_cons_deliveries = len(set(dt_route_conservative) & set(delivery_nodes))
                
            except Exception as e:
                dt_cons_success = False
                dt_cons_time = 999999
                dt_cons_distance = 999999
                dt_cons_deliveries = 0
            
            # ========================================
            # STRATEGY 3: DT with Greedy Target
            # ========================================
            greedy_target = calibration_targets[weather_severity] * 0.7  # Aggressive
            
            try:
                dt_route_greedy, _ = dt_planner.plan_route(
                    graph=network_map.G,
                    start_node=start_node,
                    delivery_nodes=delivery_nodes,
                    rain_intensity=rain_intensity,
                    target_return=greedy_target,
                    max_steps=num_deliveries,
                    temperature=0.0
                )
                
                dt_greedy_success, dt_greedy_time, dt_greedy_distance = simulate_route_execution(
                    network_map.G, dt_route_greedy, rain_intensity, 500
                )
                dt_greedy_deliveries = len(set(dt_route_greedy) & set(delivery_nodes))
                
            except Exception as e:
                dt_greedy_success = False
                dt_greedy_time = 999999
                dt_greedy_distance = 999999
                dt_greedy_deliveries = 0
            
            # ========================================
            # BASELINE: NNA
            # ========================================
            nna_route, _, _ = worker.find_disaster_aware_route(
                start_node, delivery_nodes, rain_intensity
            )
            
            if nna_route:
                nna_success, nna_time, nna_distance = simulate_route_execution(
                    network_map.G, nna_route, rain_intensity, 500
                )
                nna_deliveries = len(set(nna_route) & set(delivery_nodes))
            else:
                nna_success = False
                nna_time = 999999
                nna_distance = 999999
                nna_deliveries = 0
            
            # Store results
            results[weather_severity]['dt_aspirational_time'].append(dt_time)
            results[weather_severity]['dt_aspirational_success'].append(dt_success)
            results[weather_severity]['dt_aspirational_deliveries'].append(dt_deliveries)
            
            results[weather_severity]['dt_conservative_time'].append(dt_cons_time)
            results[weather_severity]['dt_conservative_success'].append(dt_cons_success)
            results[weather_severity]['dt_conservative_deliveries'].append(dt_cons_deliveries)
            
            results[weather_severity]['dt_greedy_time'].append(dt_greedy_time)
            results[weather_severity]['dt_greedy_success'].append(dt_greedy_success)
            results[weather_severity]['dt_greedy_deliveries'].append(dt_greedy_deliveries)
            
            results[weather_severity]['nna_time'].append(nna_time)
            results[weather_severity]['nna_success'].append(nna_success)
            results[weather_severity]['nna_deliveries'].append(nna_deliveries)
            
            # Quick summary
            if (trial * 5 + weather_severity) % 25 == 0:  # Print every 25 tests
                print(f"  Weather {weather_severity} | Rain {rain_intensity}: "
                      f"DT-Asp={dt_time:.0f}min({dt_success}) | "
                      f"DT-Con={dt_cons_time:.0f}min({dt_cons_success}) | "
                      f"DT-Grd={dt_greedy_time:.0f}min({dt_greedy_success}) | "
                      f"NNA={nna_time:.0f}min({nna_success})")
    
    # ========================================
    # COMPREHENSIVE RESULTS ANALYSIS
    # ========================================
    print("\n" + "="*70)
    print("FINAL RESULTS: RETURN-TO-GO SENSITIVITY ANALYSIS")
    print("="*70)
    
    for weather in sorted(results.keys()):
        print(f"\n{'='*70}")
        print(f"🌧️  WEATHER SEVERITY {weather}")
        print(f"{'='*70}")
        print(f"Calibrated target: {calibration_targets[weather]:.1f} min")
        
        strategies = {
            'DT (Aspirational)': 'dt_aspirational',
            'DT (Conservative)': 'dt_conservative',
            'DT (Greedy)': 'dt_greedy',
            'NNA (Baseline)': 'nna'
        }
        
        print(f"\n{'Strategy':<20} {'Success%':<12} {'Avg Time':<12} {'vs NNA':<12} {'Deliveries'}")
        print("-" * 70)
        
        nna_times = [t for t in results[weather]['nna_time'] if t < 999999]
        nna_success_rate = np.mean(results[weather]['nna_success']) * 100
        nna_avg_time = np.mean(nna_times) if nna_times else 999999
        
        for strategy_name, prefix in strategies.items():
            times = [t for t in results[weather][f'{prefix}_time'] if t < 999999]
            success_rate = np.mean(results[weather][f'{prefix}_success']) * 100
            avg_time = np.mean(times) if times else 999999
            avg_deliveries = np.mean(results[weather][f'{prefix}_deliveries'])
            
            if avg_time < 999999 and nna_avg_time < 999999:
                improvement = ((nna_avg_time - avg_time) / nna_avg_time) * 100
                vs_nna = f"{improvement:+.1f}%"
            else:
                vs_nna = "N/A"
            
            print(f"{strategy_name:<20} {success_rate:>6.1f}%     "
                  f"{avg_time:>7.1f} min  {vs_nna:<12} {avg_deliveries:.1f}/{num_deliveries}")
    
    # Overall winner
    print(f"\n{'='*70}")
    print("OVERALL BEST STRATEGY")
    print(f"{'='*70}")
    
    all_strategies_times = {}
    for prefix in ['dt_aspirational', 'dt_conservative', 'dt_greedy', 'nna']:
        all_times = []
        for weather in results.keys():
            all_times.extend([t for t in results[weather][f'{prefix}_time'] if t < 999999])
        if all_times:
            all_strategies_times[prefix] = np.mean(all_times)
    
    best_strategy = min(all_strategies_times.items(), key=lambda x: x[1])
    print(f"\n🏆 Winner: {best_strategy[0].replace('_', ' ').title()}")
    print(f"   Average time: {best_strategy[1]:.1f} minutes")
    
    for name, time in sorted(all_strategies_times.items(), key=lambda x: x[1]):
        improvement = ((all_strategies_times['nna'] - time) / all_strategies_times['nna']) * 100
        print(f"   {name.replace('_', ' ').title():<20}: {time:>6.1f} min ({improvement:+.1f}% vs NNA)")
    
    return results, calibration_targets


def simulate_route_execution(graph, route, rain_intensity, base_speed):
    """
    Simulate actual route execution with disasters
    """
    if not route or len(route) < 2:
        return False, 999999, 999999
    
    total_time = 0
    total_distance = 0
    rain_effects = {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.4, 5: 0.2}
    
    for i in range(len(route) - 1):
        from_node = route[i]
        to_node = route[i + 1]
        
        try:
            path = nx.shortest_path(graph, from_node, to_node, weight='length')
            
            for j in range(len(path) - 1):
                u, v = path[j], path[j + 1]
                
                if not graph.has_edge(u, v):
                    return False, total_time, total_distance
                
                edge_data = graph.edges[u, v]
                edge_length = edge_data.get('length', 0)
                
                # Check blocking conditions
                if edge_data.get('currently_flooded', False) and rain_intensity >= 5:
                    return False, total_time, total_distance
                if edge_data.get('currently_landslide', False) and rain_intensity >= 4:
                    return False, total_time, total_distance
                
                # Calculate speed with disaster penalties
                adjusted_speed = base_speed * rain_effects.get(rain_intensity, 0.6)
                
                if edge_data.get('currently_flooded', False):
                    adjusted_speed *= 0.5
                if edge_data.get('currently_landslide', False):
                    adjusted_speed *= 0.3
                
                if adjusted_speed <= 0:
                    return False, total_time, total_distance
                
                travel_time = edge_length / adjusted_speed
                total_time += travel_time
                total_distance += edge_length
                
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return False, total_time, total_distance
    
    return True, total_time, total_distance