"""
FIXED Filtering and Inspection Script
- Much more lenient filtering criteria
- Properly extracts total_time from aggregate_stats
- Better handling of nested data structure
"""
import pickle
import os
import json
import numpy as np
import warnings


def get_total_time(traj):
    """Extract total_time from trajectory (handles nested structure)"""
    # Try top level first
    if 'total_time' in traj and traj['total_time'] is not None:
        return traj['total_time']
    
    # Try aggregate_stats
    if 'aggregate_stats' in traj:
        stats = traj['aggregate_stats']
        if isinstance(stats, dict) and 'time' in stats:
            return stats['time']
    
    # Fallback: calculate from rewards (negative time)
    if 'total_reward' in traj:
        return abs(traj['total_reward'])
    
    return None


def filter_quality_trajectories_lenient(all_trajectories, weather_severity, target_keep_rate=0.6):
    """
    LENIENT: More permissive filtering to preserve training data
    
    Strategy:
    - Keep 60-80% of valid trajectories (not 0.1%!)
    - Weather 1-3: Keep faster deliveries (top 70%)
    - Weather 4-5: Keep all valid attempts (top 60%)
    - Always ensures diverse training data
    """
    if not all_trajectories:
        print(f"  ⚠ Weather {weather_severity}: No trajectories collected!")
        return []
    
    raw_count = len(all_trajectories)
    print(f"Weather {weather_severity}: {raw_count} raw → ", end="", flush=True)
    
    # Stage 1: Basic validity check (very lenient)
    valid = []
    for t in all_trajectories:
        # Must have core fields
        if ('states' not in t or 'actions' not in t):
            continue
        
        # Must have finite reward
        if not np.isfinite(t.get('total_reward', -np.inf)):
            continue
        
        # Must have reasonable length (at least 3 steps)
        if len(t.get('states', [])) < 3:
            continue
        
        valid.append(t)
    
    if len(valid) == 0:
        print(f"⚠ No valid trajectories!")
        return []
    
    print(f"{len(valid)} valid → ", end="", flush=True)
    
    # Stage 2: Quality filtering (lenient percentile-based)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        
        successful = [t for t in valid if t.get('success', False)]
        
        if len(successful) == 0:
            # No successes - for difficult weather, keep best attempts
            if weather_severity >= 4:
                # Keep top 50% by reward
                rewards = [t.get('total_reward', -np.inf) for t in valid]
                threshold = np.percentile(rewards, 50)
                filtered = [t for t in valid if t.get('total_reward', -np.inf) >= threshold]
                
                if len(filtered) < 5:  # Ensure minimum
                    filtered = sorted(valid, key=lambda x: x.get('total_reward', -np.inf), reverse=True)[:5]
                
                avg_reward = np.mean([t.get('total_reward', 0) for t in filtered])
                print(f"Kept {len(filtered)}/{raw_count} ({len(filtered)/raw_count*100:.1f}%, "
                      f"0 successful, avg reward: {avg_reward:.1f})")
                return filtered
            else:
                print(f"⚠ No successful trajectories in easy weather!")
                return []
        
        # Has successful trajectories
        if weather_severity >= 4:
            # Difficult weather: Keep top 60% of successful
            keep_percentile = 40  # Keep top 60%
        else:
            # Easy weather: Keep top 70% of successful
            keep_percentile = 30  # Keep top 70%
        
        # Filter by total_time (faster is better)
        times = []
        valid_successful = []
        for t in successful:
            time = get_total_time(t)
            if time is not None and np.isfinite(time):
                times.append(time)
                valid_successful.append(t)
        
        if len(valid_successful) == 0:
            # Fallback to reward-based filtering
            rewards = [t.get('total_reward', -np.inf) for t in successful]
            threshold = np.percentile(rewards, keep_percentile)
            filtered = [t for t in successful if t.get('total_reward', -np.inf) >= threshold]
        else:
            # Time-based filtering (lower is better)
            time_threshold = np.percentile(times, 100 - keep_percentile)
            filtered = []
            for t in valid_successful:
                time = get_total_time(t)
                if time <= time_threshold:
                    filtered.append(t)
        
        # Ensure minimum diversity
        if len(filtered) < 10 and len(successful) >= 10:
            # Take top 10 by time or reward
            if len(valid_successful) >= 10:
                filtered = sorted(valid_successful, key=lambda x: get_total_time(x) or float('inf'))[:10]
            else:
                filtered = sorted(successful, key=lambda x: x.get('total_reward', -np.inf), reverse=True)[:10]
    
    # Print summary
    success_count = sum(1 for t in filtered if t.get('success', False))
    avg_time = np.mean([get_total_time(t) for t in filtered if get_total_time(t) is not None]) if filtered else 0
    avg_reward = np.mean([t.get('total_reward', 0) for t in filtered]) if filtered else 0
    
    print(f"Kept {len(filtered)}/{raw_count} "
          f"({len(filtered)/raw_count*100:.1f}%, "
          f"{success_count} successful, "
          f"avg time: {avg_time:.1f}min, "
          f"avg reward: {avg_reward:.1f})")
    
    return filtered


def print_trajectory_summary(traj, idx=0):
    """Print a nice summary of a trajectory"""
    print(f"\n{'='*80}")
    print(f"Trajectory #{idx}")
    print(f"{'='*80}")
    print(f"Success: {traj.get('success', 'Unknown')}")
    print(f"Weather Severity: {traj.get('rain_intensity', traj.get('weather_severity', 'Unknown'))}")
    
    # Handle total_reward
    total_reward = traj.get('total_reward', None)
    if total_reward is not None:
        print(f"Total Reward: {total_reward:.2f}")
    else:
        print(f"Total Reward: N/A")
    
    # Handle total_time properly
    total_time = get_total_time(traj)
    if total_time is not None:
        print(f"Total Time: {total_time:.2f} minutes")
    else:
        print(f"Total Time: N/A")
    
    print(f"Steps: {len(traj.get('states', []))}")
    
    # Handle deliveries_made from aggregate_stats
    if 'aggregate_stats' in traj and isinstance(traj['aggregate_stats'], dict):
        deliveries = traj['aggregate_stats'].get('deliveries_made', 'N/A')
        print(f"Deliveries Made: {deliveries}")
    else:
        print(f"Deliveries Made: N/A")
    
    if 'states' in traj and len(traj['states']) > 0:
        print(f"\nFirst 3 states:")
        for i, state in enumerate(traj['states'][:3]):
            print(f"  Step {i}: {state}")
    
    if 'actions' in traj and len(traj['actions']) > 0:
        print(f"\nFirst 3 actions:")
        for i, action in enumerate(traj['actions'][:3]):
            print(f"  Step {i}: {action}")
    
    print(f"{'='*80}\n")


def save_trajectory_as_json(traj, filepath):
    """Save a trajectory as JSON (converts numpy arrays to lists)"""
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_traj = convert_to_serializable(traj)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_traj, f, indent=2)
    
    print(f"✓ Saved trajectory to: {filepath}")


def main():
    """Main filtering and inspection script"""
    print("="*80)
    print("TRAJECTORY FILTERING AND INSPECTION (FIXED & LENIENT)")
    print("="*80)
    
    # Configuration
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    raw_dir = os.path.join(BASE_DIR, "data", "raw")
    temp_dir = os.path.join(BASE_DIR, "data", "temp")
    processed_dir = os.path.join(BASE_DIR, "data", "processed")
    
    os.makedirs(processed_dir, exist_ok=True)
    
    raw_file = os.path.join(raw_dir, "trajectories_all_raw.pkl")
    
    print(f"\nLooking for data in:")
    print(f"  Raw directory: {raw_dir}")
    print(f"  Temp directory: {temp_dir}")
    
    all_trajectories_raw = []
    
    # Try to load raw file first
    if os.path.exists(raw_file):
        print(f"\n✓ Found raw file: {raw_file}")
        with open(raw_file, 'rb') as f:
            raw_data = pickle.load(f)
            all_trajectories_raw = raw_data.get('trajectories', [])
        print(f"  Loaded {len(all_trajectories_raw)} trajectories")
    else:
        # Try to load from temp files
        print(f"\n⚠ Raw file not found, looking for temp files...")
        if os.path.exists(temp_dir):
            temp_files = [f for f in os.listdir(temp_dir) if f.startswith('temp_cycle') and f.endswith('.pkl')]
            print(f"  Found {len(temp_files)} temp files")
            
            for temp_file in temp_files:
                temp_path = os.path.join(temp_dir, temp_file)
                try:
                    with open(temp_path, 'rb') as f:
                        temp_data = pickle.load(f)
                        
                        # Handle different formats
                        if isinstance(temp_data, list):
                            trajs = temp_data
                        elif isinstance(temp_data, dict):
                            trajs = temp_data.get('trajectories', [])
                        else:
                            print(f"    ✗ Unknown format in {temp_file}: {type(temp_data)}")
                            continue
                        
                        all_trajectories_raw.extend(trajs)
                        print(f"    ✓ Loaded {len(trajs)} from {temp_file}")
                except Exception as e:
                    print(f"    ✗ Error loading {temp_file}: {e}")
            
            print(f"\n  Total loaded from temp: {len(all_trajectories_raw)}")
        else:
            print(f"  ✗ Temp directory not found")
    
    if len(all_trajectories_raw) == 0:
        print(f"\n✗ ERROR: No trajectories found!")
        return
    
    print(f"\n{'='*80}")
    print(f"DATA LOADED: {len(all_trajectories_raw)} raw trajectories")
    print(f"{'='*80}")
    
    # Diagnostic
    if len(all_trajectories_raw) > 0:
        print(f"\n[Diagnostic] First trajectory structure:")
        first_traj = all_trajectories_raw[0]
        print(f"  Type: {type(first_traj)}")
        if isinstance(first_traj, dict):
            print(f"  Keys: {list(first_traj.keys())}")
            print(f"  Sample values:")
            for key in ['success', 'rain_intensity', 'total_reward', 'aggregate_stats']:
                if key in first_traj:
                    value = first_traj[key]
                    if isinstance(value, dict):
                        print(f"    {key}: dict with keys {list(value.keys())}")
                    else:
                        print(f"    {key}: {value}")
    
    # Group by weather
    print(f"\nGrouping by weather severity...")
    by_weather = {1: [], 2: [], 3: [], 4: [], 5: []}
    
    for traj in all_trajectories_raw:
        weather = None
        if isinstance(traj, dict):
            weather = traj.get('rain_intensity') or traj.get('weather_severity')
        if weather is None:
            weather = 3
        if weather in by_weather:
            by_weather[weather].append(traj)
    
    print(f"\nRaw distribution:")
    for w in range(1, 6):
        print(f"  Weather {w}: {len(by_weather[w])} trajectories")
    
    # Filter with LENIENT criteria
    print(f"\n{'='*80}")
    print("FILTERING FOR QUALITY (LENIENT)")
    print(f"{'='*80}\n")
    
    filtered_trajectories = []
    weather_stats = {}
    
    for weather_severity in range(1, 6):
        filtered = filter_quality_trajectories_lenient(
            by_weather[weather_severity], 
            weather_severity,
            target_keep_rate=0.6
        )
        filtered_trajectories.extend(filtered)
        
        weather_stats[weather_severity] = {
            'raw': len(by_weather[weather_severity]),
            'filtered': len(filtered),
            'success_count': sum(1 for t in filtered if t.get('success', False))
        }
    
    # Save filtered data
    print(f"\n{'='*80}")
    print("SAVING FILTERED DATA")
    print(f"{'='*80}\n")
    
    # Save as list of trajectory dicts (compatible with prepare_data)
    filtered_path = os.path.join(processed_dir, "trajectories_all.pkl")
    with open(filtered_path, "wb") as f:
        pickle.dump(filtered_trajectories, f)
    print(f"✓ Saved filtered data: {filtered_path}")
    print(f"  Count: {len(filtered_trajectories)}")
    
    # Summary
    print(f"\n{'='*80}")
    print("FILTERING SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"Total raw: {len(all_trajectories_raw)}")
    print(f"Total filtered: {len(filtered_trajectories)}")
    print(f"Filtering rate: {len(filtered_trajectories)/len(all_trajectories_raw)*100:.1f}%\n")
    
    print(f"Breakdown by weather:")
    for w in range(1, 6):
        w_stats = weather_stats[w]
        pass_rate = w_stats['filtered']/w_stats['raw']*100 if w_stats['raw'] > 0 else 0
        print(f"  Weather {w}: {w_stats['raw']:5d} → {w_stats['filtered']:5d} "
              f"({pass_rate:5.1f}%, {w_stats['success_count']} successful)")
    
    # Inspect samples
    print(f"\n{'='*80}")
    print("TRAJECTORY INSPECTION")
    print(f"{'='*80}")
    
    if len(filtered_trajectories) > 0:
        print("\n[Showing first filtered trajectory]")
        print_trajectory_summary(filtered_trajectories[0], idx=0)
        
        json_path = os.path.join(processed_dir, "sample_trajectory.json")
        save_trajectory_as_json(filtered_trajectories[0], json_path)
        
        print(f"\nSample by weather:")
        for weather in range(1, 6):
            weather_trajs = [t for t in filtered_trajectories 
                           if t.get('rain_intensity', t.get('weather_severity', 3)) == weather]
            if weather_trajs:
                t = weather_trajs[0]
                time = get_total_time(t)
                print(f"  Weather {weather}: Success={t.get('success')}, "
                      f"Reward={t.get('total_reward', 0):.1f}, "
                      f"Time={time:.1f if time else 'N/A'}min, "
                      f"Steps={len(t.get('states', []))}")
    
    print(f"\n{'='*80}")
    print("COMPLETE!")
    print(f"{'='*80}")
    print(f"\n✅ Ready for training with {len(filtered_trajectories)} diverse trajectories!")
    print(f"   Expected: ~{len(all_trajectories_raw) * 0.6:.0f} trajectories at 60% keep rate")
    print(f"   Actual: {len(filtered_trajectories)} trajectories ({len(filtered_trajectories)/len(all_trajectories_raw)*100:.1f}% kept)")


if __name__ == "__main__":
    main()