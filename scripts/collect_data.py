"""
INTEGRATED: Ultra-Diverse Data Collection with Robust Filtering

Retains your multiprocessing architecture + dynamic saving.
Adds adaptive filtering that handles failed weather conditions.
"""
import sys
import os
import pickle
import time
from multiprocessing import Process, Queue, current_process
import networkx as nx
import numpy as np
import warnings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DecisionTransformer.Utilities import NetworkMap
from SimulationWithLogging.MonteCarloWithLogging import MonteCarloSimulationWithLogging


def process_worker(G_serialized, start_node, delivery_nodes, ideal_delivery_time,
                   weather_severity, cycle_id, num_simulations, temp_save_dir, main_queue):
    """
    Worker function - runs simulations for one weather severity in one cycle
    """
    worker_name = current_process().name
    print(f"[{worker_name}] Cycle {cycle_id}, Weather {weather_severity}: Starting {num_simulations} sims")
    
    try:
        # Deserialize graph
        G = nx.node_link_graph(G_serialized, edges="links")
        
        # Create simulation with logging
        simulation = MonteCarloSimulationWithLogging(
            G,
            start_node,
            delivery_nodes,
            ideal_delivery_time,
            f"Cycle{cycle_id}-Weather{weather_severity}"
        )
        
        # Create temporary save path for this worker
        temp_save_path = os.path.join(temp_save_dir, f"temp_cycle{cycle_id}_weather{weather_severity}.pkl")
        
        # Run simulations
        trajectories = simulation.run_simulation_with_logging(
            num_simulations=num_simulations,
            weather_severity=weather_severity,
            save_path=temp_save_path
        )
        
        # Calculate statistics
        success_count = sum(1 for t in trajectories if t.get('success', False))
        success_rate = success_count / len(trajectories) if trajectories else 0
        
        result = {
            'cycle_id': cycle_id,
            'weather_severity': weather_severity,
            'trajectories': trajectories,
            'success_count': success_count,
            'success_rate': success_rate,
            'total': len(trajectories),
            'temp_file': temp_save_path
        }
        
        print(f"[{worker_name}] Cycle {cycle_id}, Weather {weather_severity}: "
              f"Complete - {success_count}/{len(trajectories)} successful ({success_rate:.1%})")
        
        main_queue.put(result)
        
    except Exception as e:
        print(f"[{worker_name}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        main_queue.put(None)


def filter_quality_trajectories_adaptive(all_trajectories, weather_severity, min_keep=3):
    """
    ROBUST: Adaptive filtering that handles failed weather conditions.
    
    Strategy:
    - Weather 1-3: Standard quality filtering
    - Weather 4-5: Very lenient, keeps learning examples even from failures
    - Always ensures minimum trajectories for training diversity
    """
    if not all_trajectories:
        print(f"  ⚠ Weather {weather_severity}: No trajectories collected!")
        return []
    
    raw_count = len(all_trajectories)
    print(f"Weather {weather_severity}: {raw_count} raw → ", end="", flush=True)
    
    # Stage 1: Filter valid trajectories (basic sanity checks)
    valid = []
    for t in all_trajectories:
        # Must have required fields and finite reward
        if ('states' not in t or 'actions' not in t or 
            not np.isfinite(t.get('total_reward', -np.inf))):
            continue
        # Must have minimum length
        if len(t.get('states', [])) < 5:
            continue
        valid.append(t)
    
    if len(valid) == 0:
        print(f"⚠ No valid trajectories!")
        return []
    
    # Stage 2: Adaptive filtering based on weather difficulty
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        
        if weather_severity >= 4:
            # DIFFICULT WEATHER: Very lenient filtering
            # Keep all attempts above minimum threshold, or best failures
            
            successful = [t for t in valid if t.get('success', False)]
            
            if len(successful) == 0:
                # No successes - keep best failures for learning!
                print(f"⚠ No successes, keeping best {min_keep} failures for learning → ", end="")
                
                # Sort by reward and take best
                rewards = [t.get('total_reward', -np.inf) for t in valid]
                n_keep = min(min_keep, len(valid))
                best_indices = np.argsort(rewards)[-n_keep:]
                filtered = [valid[i] for i in best_indices]
                
                avg_reward = np.mean([filtered[i].get('total_reward', 0) for i in range(len(filtered))])
                print(f"Kept {len(filtered)}/{raw_count} (avg reward: {avg_reward:.1f})")
                return filtered
            
            # Has some successes - keep all successful + best failures
            rewards = [t.get('total_reward', -np.inf) for t in successful]
            threshold = np.percentile(rewards, 10)  # Very lenient: top 90%
            
            filtered = [t for t in successful if t.get('total_reward', -np.inf) >= threshold]
            
            # Ensure minimum count
            if len(filtered) < min_keep:
                n_more = min_keep - len(filtered)
                # Add best unsuccessful attempts
                unsuccessful = [t for t in valid if not t.get('success', False)]
                if unsuccessful:
                    un_rewards = [t.get('total_reward', -np.inf) for t in unsuccessful]
                    best_un = np.argsort(un_rewards)[-n_more:]
                    filtered.extend([unsuccessful[i] for i in best_un])
        
        else:
            # EASIER WEATHER: Standard quality filtering
            successful = [t for t in valid if t.get('success', False)]
            
            if len(successful) == 0:
                print(f"⚠ No successful trajectories!")
                return []
            
            # Adaptive percentile based on sample size
            if len(successful) < 10:
                percentile = 25  # Keep top 75%
            elif len(successful) < 30:
                percentile = 40  # Keep top 60%
            else:
                percentile = 50  # Keep top 50%
            
            # Filter by completion time (faster is better)
            times = [t.get('total_time', float('inf')) for t in successful]
            time_threshold = np.percentile(times, 100 - percentile)  # Inverse: lower time is better
            
            filtered = [t for t in successful if t.get('total_time', float('inf')) <= time_threshold]
            
            # Ensure minimum count from successful runs
            if len(filtered) < min_keep and len(successful) >= min_keep:
                # Take best by time
                sorted_successful = sorted(successful, key=lambda x: x.get('total_time', float('inf')))
                filtered = sorted_successful[:min_keep]
    
    # Final check: ensure we have something
    if len(filtered) == 0 and len(valid) > 0:
        # Last resort: keep best N from valid
        rewards = [t.get('total_reward', -np.inf) for t in valid]
        n_keep = min(min_keep, len(valid))
        best_indices = np.argsort(rewards)[-n_keep:]
        filtered = [valid[i] for i in best_indices]
    
    # Print summary
    success_count = sum(1 for t in filtered if t.get('success', False))
    avg_time = np.mean([t.get('total_time', 0) for t in filtered]) if filtered else 0
    avg_reward = np.mean([t.get('total_reward', 0) for t in filtered]) if filtered else 0
    
    print(f"Kept {len(filtered)}/{raw_count} "
          f"({len(filtered)/raw_count*100:.0f}%, "
          f"{success_count} successful, "
          f"avg time: {avg_time:.1f}min, "
          f"avg reward: {avg_reward:.1f})")
    
    return filtered


def run_collection_cycle(cycle_id, place_query, start_node, delivery_nodes, 
                         ideal_delivery_time, num_simulations_per_weather, temp_save_dir):
    """Run ONE complete collection cycle"""
    print(f"\n{'='*80}")
    print(f"CYCLE {cycle_id}: Initializing new disaster configuration")
    print(f"{'='*80}")
    
    cycle_start = time.time()
    
    # Load network and initialize FRESH disasters
    network_map = NetworkMap(place_query)
    network_map.download_map()
    network_map.network_to_networkx()
    
    # Set fixed points
    network_map.start_node = start_node
    network_map.delivery_nodes = delivery_nodes
    network_map.ideal_delivery_time = ideal_delivery_time
    
    # Initialize DIFFERENT disasters for this cycle
    print(f"[Cycle {cycle_id}] Initializing unique disaster configuration...")
    network_map.initialize_edge_disaster_attributes()
    
    # Count disasters
    floods = sum(1 for u, v, d in network_map.G.edges(data=True) if d.get('flood_prone', 0) > 0.5)
    landslides = sum(1 for u, v, d in network_map.G.edges(data=True) if d.get('landslide_prone', 0) > 0.5)
    print(f"[Cycle {cycle_id}] Disaster profile: {floods} flood-prone, {landslides} landslide-prone edges")
    
    # Serialize graph
    graph_data = nx.node_link_data(network_map.G, edges="links")
    
    # Spawn 5 parallel workers
    print(f"[Cycle {cycle_id}] Starting 5 parallel workers (weather 1-5)...")
    
    processes = []
    main_queue = Queue()
    
    for weather_severity in range(1, 6):
        p = Process(
            target=process_worker,
            args=(
                graph_data,
                start_node,
                delivery_nodes,
                ideal_delivery_time,
                weather_severity,
                cycle_id,
                num_simulations_per_weather,
                temp_save_dir,
                main_queue
            ),
            name=f"C{cycle_id}-W{weather_severity}"
        )
        processes.append(p)
        p.start()
    
    # Collect results
    cycle_results = []
    for _ in range(5):
        try:
            result = main_queue.get(timeout=1200)
            if result is not None:
                cycle_results.append(result)
        except Exception as e:
            print(f"[Cycle {cycle_id}] ERROR collecting result: {e}")
    
    # Wait for all processes
    for p in processes:
        p.join(timeout=30)
    
    cycle_time = time.time() - cycle_start
    
    # Summary
    total_trajs = sum(r['total'] for r in cycle_results)
    total_success = sum(r['success_count'] for r in cycle_results)
    
    print(f"\n[Cycle {cycle_id}] COMPLETE:")
    print(f"  Workers completed: {len(cycle_results)}/5")
    print(f"  Total trajectories: {total_trajs}")
    print(f"  Successful: {total_success} ({total_success/total_trajs*100 if total_trajs > 0 else 0:.1f}%)")
    print(f"  Time: {cycle_time:.1f}s ({cycle_time/60:.1f} min)")
    
    return cycle_results


def main():
    """Main data collection with multiple cycles and robust filtering"""
    place_query = "La Trinidad, Benguet, Philippines"
    
    print("=" * 80)
    print("MAXIMUM DIVERSITY DATA COLLECTION + ROBUST FILTERING")
    print("Multiple Cycles + Parallel Weather Severities")
    print("=" * 80)
    
    # ========== CONFIGURATION ==========
    NUM_CYCLES = 2
    NUM_SIMULATIONS_PER_WEATHER = 10
    MIN_TRAJECTORIES_PER_WEATHER = 3  # NEW: Guarantee minimum per weather
    
    print(f"\nConfiguration:")
    print(f"  Cycles (different disaster configs): {NUM_CYCLES}")
    print(f"  Simulations per weather per cycle: {NUM_SIMULATIONS_PER_WEATHER}")
    print(f"  Weather severities per cycle: 5 (parallel)")
    print(f"  Minimum keep per weather: {MIN_TRAJECTORIES_PER_WEATHER}")
    print(f"  Total per cycle: {NUM_SIMULATIONS_PER_WEATHER * 5}")
    print(f"  GRAND TOTAL: {NUM_CYCLES * NUM_SIMULATIONS_PER_WEATHER * 5} trajectories")
    
    # ========== SETUP ==========
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(BASE_DIR, "data", "raw")
    processed_dir = os.path.join(BASE_DIR, "data", "processed")
    temp_dir = os.path.join(BASE_DIR, "data", "temp")
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    # ========== INITIAL SETUP ==========
    print("\n" + "=" * 80)
    print("Initial Setup")
    print("=" * 80)
    print("\nLoading network to determine fixed points...")
    
    network_map = NetworkMap(place_query)
    network_map.download_map()
    network_map.network_to_networkx()
    
    start_node, delivery_nodes = network_map.select_fixed_points(num_delivery_points=20)
    route, total_distance, path_sequence = network_map.nearest_neighbor_route()
    network_map.calculate_travel_time(total_distance)
    ideal_delivery_time = network_map.ideal_delivery_time
    
    print(f"  Start node: {start_node}")
    print(f"  Delivery nodes: {len(delivery_nodes)}")
    print(f"  Ideal delivery time: {ideal_delivery_time:.2f} minutes")
    
    # ========== RUN CYCLES ==========
    print("\n" + "=" * 80)
    print("Starting Multi-Cycle Data Collection")
    print("=" * 80)
    
    all_results = []
    overall_start = time.time()
    
    for cycle_id in range(NUM_CYCLES):
        print(f"\n{'█' * 80}")
        print(f"║  CYCLE {cycle_id + 1}/{NUM_CYCLES}  ║")
        print(f"{'█' * 80}")
        
        # Run one cycle
        cycle_results = run_collection_cycle(
            cycle_id=cycle_id,
            place_query=place_query,
            start_node=start_node,
            delivery_nodes=delivery_nodes,
            ideal_delivery_time=ideal_delivery_time,
            num_simulations_per_weather=NUM_SIMULATIONS_PER_WEATHER,
            temp_save_dir=temp_dir
        )
        
        all_results.extend(cycle_results)
        
        # Progress update
        progress = ((cycle_id + 1) / NUM_CYCLES) * 100
        elapsed = time.time() - overall_start
        estimated_total = elapsed / (cycle_id + 1) * NUM_CYCLES
        remaining = estimated_total - elapsed
        
        print(f"\n{'─' * 80}")
        print(f"Overall Progress: {cycle_id + 1}/{NUM_CYCLES} cycles ({progress:.0f}%)")
        print(f"  Elapsed: {elapsed/60:.1f} min")
        print(f"  Estimated remaining: {remaining/60:.1f} min")
        print(f"  Total trajectories so far: {sum(r['total'] for r in all_results)}")
        print(f"{'─' * 80}")
    
    total_time = time.time() - overall_start
    
    # ========== PROCESS RESULTS ==========
    print("\n" + "=" * 80)
    print("Processing Results")
    print("=" * 80)
    
    # Combine all trajectories
    all_trajectories_raw = []
    for result in all_results:
        if result:
            all_trajectories_raw.extend(result['trajectories'])
    
    print(f"\nRaw trajectories collected: {len(all_trajectories_raw)}")
    print(f"Total collection time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    
    # Group by weather
    by_weather = {1: [], 2: [], 3: [], 4: [], 5: []}
    for traj in all_trajectories_raw:
        weather = traj.get('rain_intensity', traj.get('weather_severity', 3))
        if weather in by_weather:
            by_weather[weather].append(traj)
    
    # Filter for quality with ROBUST filtering
    print(f"\n{'='*80}")
    print("Filtering for Quality (Adaptive Strategy)")
    print(f"{'='*80}")
    
    filtered_trajectories = []
    weather_stats = {}
    
    for weather_severity in range(1, 6):
        filtered = filter_quality_trajectories_adaptive(
            by_weather[weather_severity], 
            weather_severity,
            min_keep=MIN_TRAJECTORIES_PER_WEATHER
        )
        filtered_trajectories.extend(filtered)
        
        weather_stats[weather_severity] = {
            'raw': len(by_weather[weather_severity]),
            'filtered': len(filtered),
            'success_count': sum(1 for t in filtered if t.get('success', False))
        }
    
    # ========== SAVE ==========
    print(f"\n{'='*80}")
    print("Saving Data")
    print(f"{'='*80}\n")
    
    # Save filtered with metadata
    filtered_data = {
        'trajectories': filtered_trajectories,
        'weather_conditions': [t.get('rain_intensity', t.get('weather_severity', 3)) 
                               for t in filtered_trajectories],
        'metadata': {
            'num_cycles': NUM_CYCLES,
            'sims_per_weather_per_cycle': NUM_SIMULATIONS_PER_WEATHER,
            'min_per_weather': MIN_TRAJECTORIES_PER_WEATHER,
            'collection_time_seconds': total_time
        }
    }
    
    filtered_path = os.path.join(processed_dir, "trajectories_all.pkl")
    with open(filtered_path, "wb") as f:
        pickle.dump(filtered_data, f)
    print(f"✓ Saved filtered: {filtered_path}")
    print(f"  Count: {len(filtered_trajectories)}")
    
    # Save raw
    raw_data = {
        'trajectories': all_trajectories_raw,
        'weather_conditions': [t.get('rain_intensity', t.get('weather_severity', 3)) 
                               for t in all_trajectories_raw]
    }
    
    raw_path = os.path.join(raw_dir, "trajectories_all_raw.pkl")
    with open(raw_path, "wb") as f:
        pickle.dump(raw_data, f)
    print(f"✓ Saved raw: {raw_path}")
    print(f"  Count: {len(all_trajectories_raw)}")
    
    # Save comprehensive statistics
    stats = {
        'config': {
            'num_cycles': NUM_CYCLES,
            'sims_per_weather_per_cycle': NUM_SIMULATIONS_PER_WEATHER,
            'min_trajectories_per_weather': MIN_TRAJECTORIES_PER_WEATHER
        },
        'totals': {
            'raw': len(all_trajectories_raw),
            'filtered': len(filtered_trajectories),
            'filtering_rate': len(filtered_trajectories) / len(all_trajectories_raw) if all_trajectories_raw else 0,
            'total_time_seconds': total_time
        },
        'by_weather': weather_stats
    }
    
    stats_path = os.path.join(processed_dir, "collection_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)
    print(f"✓ Saved statistics: {stats_path}")
    
    # ========== CLEANUP TEMP FILES ==========
    print(f"\nCleaning up temporary files...")
    for result in all_results:
        if result and 'temp_file' in result:
            try:
                if os.path.exists(result['temp_file']):
                    os.remove(result['temp_file'])
            except Exception as e:
                print(f"  Warning: Could not remove {result['temp_file']}: {e}")
    print(f"✓ Cleanup complete")
    
    # ========== FINAL SUMMARY ==========
    print(f"\n{'='*80}")
    print("DATA COLLECTION COMPLETE!")
    print(f"{'='*80}")
    print(f"\nConfiguration:")
    print(f"  Cycles: {NUM_CYCLES}")
    print(f"  Simulations per weather per cycle: {NUM_SIMULATIONS_PER_WEATHER}")
    print(f"  Total raw collected: {len(all_trajectories_raw)}")
    print(f"\nResults:")
    print(f"  High-quality filtered: {len(filtered_trajectories)}")
    print(f"  Filtering rate: {len(filtered_trajectories)/len(all_trajectories_raw)*100:.1f}%")
    print(f"  Collection time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    
    print(f"\nBreakdown by weather:")
    all_weathers_have_data = True
    for w in range(1, 6):
        w_stats = weather_stats[w]
        pass_rate = w_stats['filtered']/w_stats['raw']*100 if w_stats['raw'] > 0 else 0
        print(f"  Weather {w}: {w_stats['raw']} → {w_stats['filtered']} "
              f"({pass_rate:.0f}%, {w_stats['success_count']} successful)")
        if w_stats['filtered'] == 0:
            all_weathers_have_data = False
    
    if all_weathers_have_data:
        print(f"\n✓ All weather conditions represented!")
    else:
        print(f"\n⚠ WARNING: Some weather conditions have no training data!")
        print(f"  Consider increasing NUM_SIMULATIONS_PER_WEATHER or lowering MIN_TRAJECTORIES_PER_WEATHER")
    
    print(f"\n✓ Ready for training with {len(filtered_trajectories)} diverse trajectories!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()