"""
Trajectory Data Inspector
Diagnoses what's wrong with collected trajectories
"""
import pickle
import numpy as np
from pathlib import Path


def inspect_raw_trajectories(raw_data_path):
    """
    Thoroughly inspect raw trajectory data to find issues
    """
    print("="*80)
    print("TRAJECTORY DATA INSPECTION")
    print("="*80)
    
    # Load raw data
    print(f"\nLoading: {raw_data_path}")
    with open(raw_data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Handle different data structures
    if isinstance(data, dict):
        trajectories = data.get('trajectories', [])
        print(f"Loaded: {len(trajectories)} trajectories from dict")
    elif isinstance(data, list):
        trajectories = data
        print(f"Loaded: {len(trajectories)} trajectories from list")
    else:
        print(f"ERROR: Unexpected data type: {type(data)}")
        return
    
    if len(trajectories) == 0:
        print("ERROR: No trajectories found!")
        return
    
    print("\n" + "="*80)
    print("SAMPLE TRAJECTORY ANALYSIS")
    print("="*80)
    
    # Inspect first 3 trajectories
    for i in range(min(3, len(trajectories))):
        print(f"\n--- Trajectory {i} ---")
        traj = trajectories[i]
        
        print(f"Type: {type(traj)}")
        
        if isinstance(traj, dict):
            print(f"Keys: {list(traj.keys())}")
            
            # Check each critical field
            for key in ['states', 'actions', 'rewards', 'total_reward', 
                       'success', 'total_time', 'weather_severity', 'rain_intensity']:
                if key in traj:
                    value = traj[key]
                    if isinstance(value, (list, np.ndarray)):
                        print(f"  {key}: {type(value).__name__} length={len(value)}")
                        if len(value) > 0:
                            print(f"    First element: {value[0]}")
                    else:
                        print(f"  {key}: {value}")
                else:
                    print(f"  {key}: MISSING ❌")
        else:
            print(f"  ERROR: Expected dict, got {type(traj)}")
    
    print("\n" + "="*80)
    print("FIELD PRESENCE STATISTICS")
    print("="*80)
    
    # Count field presence across all trajectories
    field_counts = {}
    critical_fields = ['states', 'actions', 'rewards', 'total_reward', 
                      'success', 'total_time', 'weather_severity']
    
    for field in critical_fields:
        count = sum(1 for t in trajectories if isinstance(t, dict) and field in t)
        field_counts[field] = count
        percentage = (count / len(trajectories)) * 100
        status = "✓" if percentage == 100 else "⚠" if percentage > 50 else "❌"
        print(f"{status} {field}: {count}/{len(trajectories)} ({percentage:.1f}%)")
    
    print("\n" + "="*80)
    print("DATA QUALITY CHECKS")
    print("="*80)
    
    # Check data quality
    valid_count = 0
    has_states = 0
    has_actions = 0
    has_finite_reward = 0
    min_length = 0
    
    for traj in trajectories:
        if not isinstance(traj, dict):
            continue
        
        # Check states
        if 'states' in traj and len(traj['states']) > 0:
            has_states += 1
            if len(traj['states']) >= 5:
                min_length += 1
        
        # Check actions
        if 'actions' in traj and len(traj['actions']) > 0:
            has_actions += 1
        
        # Check reward
        reward = traj.get('total_reward', -np.inf)
        if np.isfinite(reward):
            has_finite_reward += 1
        
        # Would pass validation?
        if ('states' in traj and len(traj.get('states', [])) >= 5 and
            'actions' in traj and len(traj.get('actions', [])) > 0 and
            np.isfinite(traj.get('total_reward', -np.inf))):
            valid_count += 1
    
    print(f"Has 'states' with data: {has_states}/{len(trajectories)} ({has_states/len(trajectories)*100:.1f}%)")
    print(f"Has 'actions' with data: {has_actions}/{len(trajectories)} ({has_actions/len(trajectories)*100:.1f}%)")
    print(f"Has finite reward: {has_finite_reward}/{len(trajectories)} ({has_finite_reward/len(trajectories)*100:.1f}%)")
    print(f"Meets min length (≥5): {min_length}/{len(trajectories)} ({min_length/len(trajectories)*100:.1f}%)")
    print(f"\n✓ WOULD PASS FILTERING: {valid_count}/{len(trajectories)} ({valid_count/len(trajectories)*100:.1f}%)")
    
    print("\n" + "="*80)
    print("REWARD DISTRIBUTION")
    print("="*80)
    
    # Analyze rewards
    rewards = []
    for traj in trajectories:
        if isinstance(traj, dict):
            r = traj.get('total_reward', None)
            if r is not None:
                rewards.append(r)
    
    if rewards:
        rewards = np.array(rewards)
        finite_rewards = rewards[np.isfinite(rewards)]
        
        print(f"Total rewards found: {len(rewards)}")
        print(f"Finite rewards: {len(finite_rewards)}")
        print(f"NaN/Inf rewards: {len(rewards) - len(finite_rewards)}")
        
        if len(finite_rewards) > 0:
            print(f"\nFinite reward statistics:")
            print(f"  Min: {np.min(finite_rewards):.2f}")
            print(f"  Max: {np.max(finite_rewards):.2f}")
            print(f"  Mean: {np.mean(finite_rewards):.2f}")
            print(f"  Median: {np.median(finite_rewards):.2f}")
    else:
        print("No rewards found!")
    
    print("\n" + "="*80)
    print("SUCCESS RATE")
    print("="*80)
    
    success_count = sum(1 for t in trajectories 
                       if isinstance(t, dict) and t.get('success', False))
    print(f"Successful trajectories: {success_count}/{len(trajectories)} ({success_count/len(trajectories)*100:.1f}%)")
    
    print("\n" + "="*80)
    print("DIAGNOSIS")
    print("="*80)
    
    if valid_count == 0:
        print("\n❌ CRITICAL: No valid trajectories!")
        print("\nPossible causes:")
        print("1. MonteCarloSimulationWithLogging is not populating 'states' and 'actions'")
        print("2. Trajectory structure doesn't match expected format")
        print("3. Simulation is crashing before saving data")
        print("\n👉 ACTION REQUIRED:")
        print("   Check MonteCarloSimulationWithLogging.run_simulation_with_logging()")
        print("   Ensure it returns dicts with 'states', 'actions', and 'total_reward'")
    elif valid_count < len(trajectories) * 0.5:
        print("\n⚠ WARNING: Most trajectories are invalid!")
        print(f"   {len(trajectories) - valid_count} trajectories missing critical fields")
    else:
        print("\n✓ Data looks mostly good!")
        print(f"   {valid_count} trajectories should pass filtering")
    
    print("\n" + "="*80)
    print("RECOMMENDED FIXES")
    print("="*80)
    
    if has_states < len(trajectories) * 0.9:
        print("\n1. Fix 'states' collection:")
        print("   - Verify MonteCarloSimulationWithLogging stores state history")
        print("   - Check that states are appended during simulation steps")
    
    if has_actions < len(trajectories) * 0.9:
        print("\n2. Fix 'actions' collection:")
        print("   - Verify action logging in simulation")
        print("   - Ensure actions list is populated")
    
    if has_finite_reward < len(trajectories) * 0.9:
        print("\n3. Fix reward calculation:")
        print("   - Check for division by zero")
        print("   - Verify reward calculation doesn't produce NaN/Inf")
        print("   - Add reward = max(reward, -1e6) bounds")


if __name__ == "__main__":
    # Adjust path to your raw data
    RAW_DATA = r"C:\Users\Acer Nitro\Documents\CSC FILES\4th Year First Semester\Intellegent Systems\Transformer-Based Last Mile Logistics\data\raw\trajectories_all_raw.pkl"
    
    inspect_raw_trajectories(RAW_DATA)