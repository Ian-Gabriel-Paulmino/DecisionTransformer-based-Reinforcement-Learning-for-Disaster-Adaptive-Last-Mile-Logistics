"""
NO-FILTER Data Collection - Keep EVERYTHING
Maximum data for maximum learning
"""
import pickle
import os
import json
import numpy as np


def main():
    """Load ALL trajectories without any filtering"""
    print("="*80)
    print("LOADING ALL TRAJECTORIES (NO FILTERING)")
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
    
    all_trajectories = []
    
    # Try raw file first
    if os.path.exists(raw_file):
        print(f"\n✓ Found raw file: {raw_file}")
        with open(raw_file, 'rb') as f:
            raw_data = pickle.load(f)
            all_trajectories = raw_data.get('trajectories', [])
        print(f"  Loaded {len(all_trajectories)} trajectories")
    else:
        # Load from temp files
        print(f"\n⚠ Raw file not found, loading from temp files...")
        if os.path.exists(temp_dir):
            temp_files = sorted([f for f in os.listdir(temp_dir) 
                               if f.startswith('temp_cycle') and f.endswith('.pkl')])
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
                            print(f"    ✗ Unknown format in {temp_file}")
                            continue
                        
                        all_trajectories.extend(trajs)
                        print(f"    ✓ Loaded {len(trajs)} from {temp_file}")
                except Exception as e:
                    print(f"    ✗ Error loading {temp_file}: {e}")
            
            print(f"\n  Total loaded: {len(all_trajectories)}")
    
    if len(all_trajectories) == 0:
        print(f"\n✗ ERROR: No trajectories found!")
        return
    
    print(f"\n{'='*80}")
    print(f"LOADED: {len(all_trajectories)} trajectories")
    print(f"{'='*80}")
    
    # Basic validation only - remove completely broken trajectories
    print(f"\nApplying MINIMAL validation (only remove broken data)...")
    valid_trajectories = []
    
    for t in all_trajectories:
        # Must have basic structure
        if not isinstance(t, dict):
            continue
        if 'states' not in t or 'actions' not in t:
            continue
        if len(t.get('states', [])) < 2:  # At least 2 steps
            continue
        if not np.isfinite(t.get('total_reward', -np.inf)):
            continue
        
        valid_trajectories.append(t)
    
    removed = len(all_trajectories) - len(valid_trajectories)
    print(f"  Removed {removed} broken trajectories ({removed/len(all_trajectories)*100:.1f}%)")
    print(f"  Kept {len(valid_trajectories)} trajectories ({len(valid_trajectories)/len(all_trajectories)*100:.1f}%)")
    
    # Group by weather for stats
    by_weather = {1: [], 2: [], 3: [], 4: [], 5: []}
    for traj in valid_trajectories:
        weather = traj.get('rain_intensity', traj.get('weather_severity', 3))
        if weather in by_weather:
            by_weather[weather].append(traj)
    
    print(f"\nDistribution by weather:")
    for w in range(1, 6):
        success_count = sum(1 for t in by_weather[w] if t.get('success', False))
        success_rate = success_count / len(by_weather[w]) * 100 if by_weather[w] else 0
        print(f"  Weather {w}: {len(by_weather[w]):5d} trajectories "
              f"({success_count} successful, {success_rate:.1f}%)")
    
    # Save ALL valid trajectories
    print(f"\n{'='*80}")
    print("SAVING ALL TRAJECTORIES")
    print(f"{'='*80}\n")
    
    output_path = os.path.join(processed_dir, "trajectories_all.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(valid_trajectories, f)
    
    print(f"✓ Saved {len(valid_trajectories)} trajectories to:")
    print(f"  {output_path}")
    
    # Save metadata
    metadata = {
        'total_trajectories': len(valid_trajectories),
        'by_weather': {w: len(by_weather[w]) for w in range(1, 6)},
        'success_by_weather': {
            w: sum(1 for t in by_weather[w] if t.get('success', False))
            for w in range(1, 6)
        }
    }
    
    meta_path = os.path.join(processed_dir, "data_stats.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to: {meta_path}")
    
    print(f"\n{'='*80}")
    print("COMPLETE!")
    print(f"{'='*80}")
    print(f"\n✅ Ready for training with {len(valid_trajectories)} trajectories!")
    print(f"   NO FILTERING - Maximum data for maximum learning")
    print(f"   All weather conditions: {sum(len(by_weather[w]) > 0 for w in range(1, 6))}/5")
    print(f"\nNext steps:")
    print(f"  1. Use larger model config (64-96 hidden size)")
    print(f"  2. Train on GPU if possible")
    print(f"  3. Train for 100+ epochs")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()