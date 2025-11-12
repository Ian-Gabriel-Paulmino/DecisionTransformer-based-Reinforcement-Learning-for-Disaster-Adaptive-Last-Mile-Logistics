"""
Data Quality Analyzer for Decision Transformer Training Data

Diagnoses issues with collected trajectories:
- Reward distributions
- Time distributions
- Success rates
- Data balance across weather conditions
"""

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json


def load_all_temp_files(temp_dir):
    """Load all trajectories from temp files"""
    print("="*80)
    print("LOADING DATA FROM TEMP FILES")
    print("="*80)
    
    temp_files = sorted([f for f in os.listdir(temp_dir) 
                        if f.startswith('temp_cycle') and f.endswith('.pkl')])
    
    print(f"\nFound {len(temp_files)} temp files:")
    for f in temp_files:
        print(f"  - {f}")
    
    all_trajectories = []
    
    for temp_file in temp_files:
        temp_path = os.path.join(temp_dir, temp_file)
        try:
            with open(temp_path, 'rb') as f:
                data = pickle.load(f)
                
                # Handle different formats
                if isinstance(data, list):
                    trajs = data
                elif isinstance(data, dict):
                    trajs = data.get('trajectories', [])
                else:
                    print(f"  ⚠ Unknown format in {temp_file}")
                    continue
                
                all_trajectories.extend(trajs)
                print(f"  ✓ Loaded {len(trajs)} from {temp_file}")
        except Exception as e:
            print(f"  ✗ Error loading {temp_file}: {e}")
    
    print(f"\nTotal trajectories loaded: {len(all_trajectories)}")
    return all_trajectories


def analyze_trajectory_quality(trajectories):
    """Comprehensive quality analysis"""
    print("\n" + "="*80)
    print("DATA QUALITY ANALYSIS")
    print("="*80)
    
    # Group by weather
    by_weather = defaultdict(list)
    for traj in trajectories:
        weather = traj.get('rain_intensity', traj.get('weather_severity', 3))
        by_weather[weather].append(traj)
    
    # Statistics storage
    stats = {
        'overall': {},
        'by_weather': {}
    }
    
    # Overall statistics
    print(f"\n📊 OVERALL STATISTICS")
    print("-" * 80)
    
    total = len(trajectories)
    success_count = sum(1 for t in trajectories if t.get('success', False))
    success_rate = success_count / total * 100 if total > 0 else 0
    
    print(f"Total trajectories: {total}")
    print(f"Successful: {success_count} ({success_rate:.1f}%)")
    print(f"Failed: {total - success_count} ({100-success_rate:.1f}%)")
    
    stats['overall']['total'] = total
    stats['overall']['success_count'] = success_count
    stats['overall']['success_rate'] = success_rate
    
    # Reward analysis
    print(f"\n💰 REWARD ANALYSIS")
    print("-" * 80)
    
    all_rewards = [t.get('total_reward', 0) for t in trajectories if np.isfinite(t.get('total_reward', -np.inf))]
    
    if all_rewards:
        print(f"Reward statistics:")
        print(f"  Min: {min(all_rewards):.1f}")
        print(f"  25th percentile: {np.percentile(all_rewards, 25):.1f}")
        print(f"  Median: {np.median(all_rewards):.1f}")
        print(f"  75th percentile: {np.percentile(all_rewards, 75):.1f}")
        print(f"  Max: {max(all_rewards):.1f}")
        print(f"  Mean: {np.mean(all_rewards):.1f}")
        print(f"  Std: {np.std(all_rewards):.1f}")
        
        stats['overall']['reward_stats'] = {
            'min': float(min(all_rewards)),
            'q25': float(np.percentile(all_rewards, 25)),
            'median': float(np.median(all_rewards)),
            'q75': float(np.percentile(all_rewards, 75)),
            'max': float(max(all_rewards)),
            'mean': float(np.mean(all_rewards)),
            'std': float(np.std(all_rewards))
        }
    
    # Time analysis
    print(f"\n⏱️  TIME ANALYSIS")
    print("-" * 80)
    
    all_times = [t.get('aggregate_stats', {}).get('time', 0) 
                 for t in trajectories 
                 if t.get('aggregate_stats', {}).get('time', 0) > 0]
    
    if all_times:
        print(f"Time statistics (minutes):")
        print(f"  Min: {min(all_times):.1f}")
        print(f"  25th percentile: {np.percentile(all_times, 25):.1f}")
        print(f"  Median: {np.median(all_times):.1f}")
        print(f"  75th percentile: {np.percentile(all_times, 75):.1f}")
        print(f"  Max: {max(all_times):.1f}")
        print(f"  Mean: {np.mean(all_times):.1f}")
        print(f"  Std: {np.std(all_times):.1f}")
        
        # Convert to actual times (negative rewards)
        print(f"\n  Reward equivalents (negative time):")
        print(f"    Best reward (fastest): {-min(all_times):.1f}")
        print(f"    Median reward: {-np.median(all_times):.1f}")
        print(f"    Worst reward (slowest): {-max(all_times):.1f}")
        
        stats['overall']['time_stats'] = {
            'min': float(min(all_times)),
            'q25': float(np.percentile(all_times, 25)),
            'median': float(np.median(all_times)),
            'q75': float(np.percentile(all_times, 75)),
            'max': float(max(all_times)),
            'mean': float(np.mean(all_times)),
            'std': float(np.std(all_times))
        }
    
    # Weather-stratified analysis
    print(f"\n🌧️  WEATHER-STRATIFIED ANALYSIS")
    print("-" * 80)
    
    for weather in sorted(by_weather.keys()):
        weather_trajs = by_weather[weather]
        print(f"\nWeather Severity {weather}:")
        print(f"  Count: {len(weather_trajs)}")
        
        success = sum(1 for t in weather_trajs if t.get('success', False))
        success_rate = success / len(weather_trajs) * 100 if weather_trajs else 0
        print(f"  Success rate: {success}/{len(weather_trajs)} ({success_rate:.1f}%)")
        
        # Rewards for this weather
        weather_rewards = [t.get('total_reward', 0) for t in weather_trajs 
                          if np.isfinite(t.get('total_reward', -np.inf))]
        
        if weather_rewards:
            print(f"  Reward range: {min(weather_rewards):.1f} to {max(weather_rewards):.1f}")
            print(f"  Reward median: {np.median(weather_rewards):.1f}")
        
        # Times for this weather
        weather_times = [t.get('aggregate_stats', {}).get('time', 0) 
                        for t in weather_trajs 
                        if t.get('aggregate_stats', {}).get('time', 0) > 0]
        
        if weather_times:
            print(f"  Time range: {min(weather_times):.1f} to {max(weather_times):.1f} min")
            print(f"  Time median: {np.median(weather_times):.1f} min")
            print(f"  Time mean: {np.mean(weather_times):.1f} min")
        
        # Trajectory lengths
        traj_lengths = [len(t.get('states', [])) for t in weather_trajs]
        if traj_lengths:
            print(f"  Trajectory lengths: {min(traj_lengths)} to {max(traj_lengths)} steps")
            print(f"  Average length: {np.mean(traj_lengths):.1f} steps")
        
        stats['by_weather'][weather] = {
            'count': len(weather_trajs),
            'success_count': success,
            'success_rate': success_rate,
            'reward_median': float(np.median(weather_rewards)) if weather_rewards else None,
            'time_median': float(np.median(weather_times)) if weather_times else None,
            'time_mean': float(np.mean(weather_times)) if weather_times else None,
            'avg_trajectory_length': float(np.mean(traj_lengths)) if traj_lengths else None
        }
    
    return stats, by_weather


def diagnose_problems(trajectories, stats):
    """Identify specific problems with the data"""
    print("\n" + "="*80)
    print("🔍 PROBLEM DIAGNOSIS")
    print("="*80)
    
    problems = []
    warnings = []
    
    # Check 1: Data imbalance
    print(f"\n1. Data Balance Check")
    print("-" * 80)
    
    by_weather = defaultdict(list)
    for traj in trajectories:
        weather = traj.get('rain_intensity', traj.get('weather_severity', 3))
        by_weather[weather].append(traj)
    
    counts = [len(by_weather[w]) for w in range(1, 6)]
    min_count = min(counts)
    max_count = max(counts)
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print(f"Weather distribution:")
    for w in range(1, 6):
        count = len(by_weather[w])
        pct = count / len(trajectories) * 100 if trajectories else 0
        print(f"  Weather {w}: {count:4d} ({pct:5.1f}%)")
    
    if imbalance_ratio > 3:
        problem = f"❌ SEVERE IMBALANCE: {imbalance_ratio:.1f}x ratio between weather conditions"
        print(f"\n{problem}")
        problems.append(problem)
    elif imbalance_ratio > 2:
        warning = f"⚠️  MODERATE IMBALANCE: {imbalance_ratio:.1f}x ratio"
        print(f"\n{warning}")
        warnings.append(warning)
    else:
        print(f"\n✓ Good balance (ratio: {imbalance_ratio:.1f}x)")
    
    # Check 2: Success rate analysis
    print(f"\n2. Success Rate Check")
    print("-" * 80)
    
    overall_success = stats['overall']['success_rate']
    print(f"Overall success rate: {overall_success:.1f}%")
    
    if overall_success < 30:
        problem = f"❌ VERY LOW SUCCESS RATE: {overall_success:.1f}%"
        print(f"{problem}")
        problems.append(problem)
        print(f"   Most trajectories are failures. Model will learn to fail!")
    elif overall_success < 50:
        warning = f"⚠️  LOW SUCCESS RATE: {overall_success:.1f}%"
        print(f"{warning}")
        warnings.append(warning)
        print(f"   Model may learn conservative/failing strategies")
    else:
        print(f"✓ Adequate success rate")
    
    # Check 3: Reward distribution
    print(f"\n3. Reward Distribution Check")
    print("-" * 80)
    
    if 'reward_stats' in stats['overall']:
        reward_median = stats['overall']['reward_stats']['median']
        reward_std = stats['overall']['reward_stats']['std']
        reward_range = stats['overall']['reward_stats']['max'] - stats['overall']['reward_stats']['min']
        
        print(f"Median reward: {reward_median:.1f}")
        print(f"Std deviation: {reward_std:.1f}")
        print(f"Range: {reward_range:.1f}")
        
        # Check if distribution is too narrow
        if reward_std < 20:
            problem = f"❌ VERY NARROW DISTRIBUTION: std={reward_std:.1f}"
            print(f"\n{problem}")
            problems.append(problem)
            print(f"   All trajectories too similar. Model won't learn to optimize!")
        elif reward_std < 40:
            warning = f"⚠️  NARROW DISTRIBUTION: std={reward_std:.1f}"
            print(f"\n{warning}")
            warnings.append(warning)
            print(f"   Limited diversity in trajectory quality")
        else:
            print(f"\n✓ Good reward diversity")
        
        # Check if rewards are too negative (slow routes)
        if reward_median < -150:
            problem = f"❌ REWARDS TOO NEGATIVE: median={reward_median:.1f}"
            print(f"\n{problem}")
            problems.append(problem)
            print(f"   Training data consists of SLOW routes (>150 min)")
            print(f"   Model learned to be slow!")
        elif reward_median < -130:
            warning = f"⚠️  REWARDS FAIRLY NEGATIVE: median={reward_median:.1f}"
            print(f"\n{warning}")
            warnings.append(warning)
            print(f"   Training data slightly slower than optimal")
    
    # Check 4: Time distribution analysis
    print(f"\n4. Time Distribution Check")
    print("-" * 80)
    
    if 'time_stats' in stats['overall']:
        time_median = stats['overall']['time_stats']['median']
        time_mean = stats['overall']['time_stats']['mean']
        time_std = stats['overall']['time_stats']['std']
        
        print(f"Median time: {time_median:.1f} min")
        print(f"Mean time: {time_mean:.1f} min")
        print(f"Std deviation: {time_std:.1f} min")
        
        # Compare to your evaluation results
        print(f"\nComparison to NNA baseline:")
        print(f"  NNA achieves: ~130 min in Weather 1")
        print(f"  Your training data median: {time_median:.1f} min")
        
        if time_median > 140:
            problem = f"❌ TRAINING DATA TOO SLOW: {time_median:.1f} min"
            print(f"\n{problem}")
            problems.append(problem)
            print(f"   Model trained on slower routes than NNA can achieve!")
        elif time_median > 135:
            warning = f"⚠️  TRAINING DATA SLIGHTLY SLOW: {time_median:.1f} min"
            print(f"\n{warning}")
            warnings.append(warning)
    
    # Check 5: Weather-specific issues
    print(f"\n5. Weather-Specific Check")
    print("-" * 80)
    
    for weather in sorted(stats['by_weather'].keys()):
        w_stats = stats['by_weather'][weather]
        print(f"\nWeather {weather}:")
        
        if w_stats['count'] < 100:
            warning = f"  ⚠️  Only {w_stats['count']} examples (need 500+)"
            print(warning)
            warnings.append(f"Weather {weather}: " + warning)
        
        if w_stats['success_rate'] < 20 and weather < 4:
            problem = f"  ❌ Success rate too low for Weather {weather}: {w_stats['success_rate']:.1f}%"
            print(problem)
            problems.append(problem)
    
    # Summary
    print(f"\n" + "="*80)
    print("DIAGNOSIS SUMMARY")
    print("="*80)
    
    if not problems and not warnings:
        print(f"\n✅ DATA QUALITY: GOOD")
        print(f"   No critical issues found!")
    else:
        print(f"\n🚨 CRITICAL PROBLEMS: {len(problems)}")
        for p in problems:
            print(f"   {p}")
        
        print(f"\n⚠️  WARNINGS: {len(warnings)}")
        for w in warnings[:5]:  # Show first 5
            print(f"   {w}")
        if len(warnings) > 5:
            print(f"   ... and {len(warnings)-5} more")
    
    return problems, warnings


def generate_visualizations(trajectories, by_weather, output_dir):
    """Create diagnostic visualizations"""
    print(f"\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Figure 1: Reward distributions by weather
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Reward Distributions by Weather Severity', fontsize=16)
    
    for idx, weather in enumerate(sorted(by_weather.keys())):
        ax = axes[idx // 3, idx % 3]
        
        weather_rewards = [t.get('total_reward', 0) for t in by_weather[weather]
                          if np.isfinite(t.get('total_reward', -np.inf))]
        
        if weather_rewards:
            ax.hist(weather_rewards, bins=30, alpha=0.7, color='blue', edgecolor='black')
            ax.axvline(np.median(weather_rewards), color='red', linestyle='--', 
                      label=f'Median: {np.median(weather_rewards):.1f}')
            ax.set_xlabel('Total Reward (negative minutes)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Weather {weather} (n={len(weather_rewards)})')
            ax.legend()
            ax.grid(alpha=0.3)
    
    # Hide unused subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'reward_distributions.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {fig_path}")
    plt.close()
    
    # Figure 2: Time distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Time Distributions by Weather Severity', fontsize=16)
    
    for idx, weather in enumerate(sorted(by_weather.keys())):
        ax = axes[idx // 3, idx % 3]
        
        weather_times = [t.get('aggregate_stats', {}).get('time', 0) 
                        for t in by_weather[weather]
                        if t.get('aggregate_stats', {}).get('time', 0) > 0]
        
        if weather_times:
            ax.hist(weather_times, bins=30, alpha=0.7, color='green', edgecolor='black')
            ax.axvline(np.median(weather_times), color='red', linestyle='--',
                      label=f'Median: {np.median(weather_times):.1f} min')
            ax.axvline(130, color='orange', linestyle=':', linewidth=2,
                      label='NNA baseline (~130 min)')
            ax.set_xlabel('Delivery Time (minutes)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Weather {weather} (n={len(weather_times)})')
            ax.legend()
            ax.grid(alpha=0.3)
    
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'time_distributions.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {fig_path}")
    plt.close()
    
    # Figure 3: Success rates by weather
    fig, ax = plt.subplots(figsize=(10, 6))
    
    weathers = sorted(by_weather.keys())
    success_rates = [
        sum(1 for t in by_weather[w] if t.get('success', False)) / len(by_weather[w]) * 100
        for w in weathers
    ]
    
    bars = ax.bar(weathers, success_rates, color='steelblue', edgecolor='black')
    ax.set_xlabel('Weather Severity', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Success Rates Across Weather Conditions', fontsize=14)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'success_rates.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {fig_path}")
    plt.close()
    
    print(f"\n✓ All visualizations saved to: {output_dir}")


def convert_to_json_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {str(k): convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def main():
    """Main analysis function"""
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    temp_dir = os.path.join(BASE_DIR, "data", "temp")
    output_dir = os.path.join(BASE_DIR, "analysis_results")
    
    # Load data
    trajectories = load_all_temp_files(temp_dir)
    
    if len(trajectories) == 0:
        print("\n❌ ERROR: No trajectories found!")
        return
    
    # Analyze quality
    stats, by_weather = analyze_trajectory_quality(trajectories)
    
    # Diagnose problems
    problems, warnings = diagnose_problems(trajectories, stats)
    
    # Generate visualizations
    generate_visualizations(trajectories, by_weather, output_dir)
    
    # Save detailed report
    report_path = os.path.join(output_dir, 'data_quality_report.json')
    
    def convert_to_json_serializable(obj):
        """Recursively convert numpy types to native Python types for JSON"""
        if isinstance(obj, dict):
            return {str(k): convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        else:
            return obj
    
    report = {
        'statistics': stats,
        'problems': problems,
        'warnings': warnings,
        'total_trajectories': len(trajectories),
        'weather_distribution': {str(w): len(by_weather[w]) for w in sorted(by_weather.keys())}
    }
    
    # Convert numpy types to native Python types
    report = convert_to_json_serializable(report)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Detailed report saved to: {report_path}")
    
    # Final recommendations
    print(f"\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if len(problems) > 0:
        print(f"\n🚨 CRITICAL ACTIONS REQUIRED:")
        print(f"\n1. Your data has significant quality issues")
        print(f"2. Current model trained on this data will perform poorly")
        print(f"3. Recommended: Collect NEW data with these fixes:")
        print(f"   a) Increase simulations per weather to 1000+")
        print(f"   b) Don't filter by 'quality' - keep ALL successful routes")
        print(f"   c) Especially keep FAST routes (time < 130 min)")
        print(f"   d) Balance data across weather conditions")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()