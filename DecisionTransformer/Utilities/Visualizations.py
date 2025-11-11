import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np









def plot_deliveries_per_rain_intensity_combined(simulation_runs):
    """
    Plot deliveries per rain intensity level for multiple simulation runs on the same graph
    """
    plt.figure(figsize=(10, 6))
    
    bar_width = 0.15
    index = np.arange(5) + 1  # Rain levels 1-5
    
    for i, simulation in enumerate(simulation_runs):
        rain_intensities = [result['rain_intensity'] for result in simulation.simulation_results['all_runs']]
        rain_counts = [rain_intensities.count(level) for level in range(1, 6)]
        
        offset = bar_width * (i - len(simulation_runs)/2 + 0.5)
        plt.bar(index + offset, rain_counts, bar_width, 
                label=f'Simulation with Weather Severity {i+1}', 
                alpha=0.7)
    
    plt.xlabel('Rain intensity')
    plt.ylabel('Number of Delivery Attempts')
    plt.title('Delivery Attempts per rain intensity Across Simulation Runs')
    plt.xticks(index)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()




def visualize_combined_simulation_results(simulation_runs):
    """
    Create comprehensive visualizations for multiple delivery simulation runs,
    with all results superimposed on the same graphs
    """
    # Convert all results to a single DataFrame with a 'run' column
    all_data = []
    for i, simulation in enumerate(simulation_runs):
        results = simulation.simulation_results['all_runs']
        
        for result in results:
            all_data.append({
                'run': i+1,
                'time': result['time'] if result['success'] else None,
                'success': result['success'],
                'status': 'Success' if result['success'] else 'Failure',
                'reason': result.get('reason', None) if not result['success'] else None,
                'rain_intensity': result['rain_intensity']
            })
    
    df = pd.DataFrame(all_data)
    
    # Print summary statistics for each run
    for i, simulation in enumerate(simulation_runs):
        results = simulation.simulation_results['all_runs']
        successful = [result for result in results if result['success']]
        failed = [result for result in results if not result['success']]
        success_times = [result['time'] for result in successful]
        
        print(f"\nRun {i+1} Summary:")
        print(f"  Total runs: {len(results)}")
        print(f"  Successful deliveries: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
        print(f"  Failed deliveries: {len(failed)} ({len(failed)/len(results)*100:.1f}%)")
        if success_times:
            print(f"  Average delivery time (successful): {sum(success_times)/len(success_times):.2f} minutes")
            print(f"  Fastest delivery: {min(success_times):.2f} minutes")
            print(f"  Slowest delivery: {max(success_times):.2f} minutes")
    
    # 1. Success vs Failure Bar Chart with percentages
    plt.figure(figsize=(12, 7))
    
    # Count successes and failures for each run
    success_counts = []
    failure_counts = []
    
    for i in range(1, len(simulation_runs) + 1):
        run_data = df[df['run'] == i]
        success_counts.append(sum(run_data['success']))
        failure_counts.append(sum(~run_data['success']))
    
    # Create grouped bar chart
    bar_width = 0.35
    index = np.arange(len(simulation_runs))
    
    plt.bar(index, success_counts, bar_width, label='Success', color='forestgreen', alpha=0.7)
    plt.bar(index, failure_counts, bar_width, bottom=success_counts, label='Failure', color='firebrick', alpha=0.7)
    
    # Add percentage labels
    for i in range(len(simulation_runs)):
        total = success_counts[i] + failure_counts[i]
        success_pct = (success_counts[i] / total) * 100
        
        # Success label
        plt.text(i, success_counts[i]/2, f"{success_counts[i]}\n({success_pct:.1f}%)", 
                 ha='center', va='center', color='white', fontweight='bold')
        
        # Failure label
        if failure_counts[i] > 0:
            plt.text(i, success_counts[i] + failure_counts[i]/2, 
                    f"{failure_counts[i]}\n({100-success_pct:.1f}%)", 
                    ha='center', va='center', color='white', fontweight='bold')
    
    plt.title('Delivery Outcomes Across Simulation Runs', fontsize=14)
    plt.xlabel('Simulation Run', fontsize=12)
    plt.ylabel('Number of Deliveries', fontsize=12)
    plt.xticks(index, [f'Simulation with Weather Severity {i+1}' for i in range(len(simulation_runs))])
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # 2. Distribution of Successful Delivery Times
    plt.figure(figsize=(12, 7))
    
    for i, simulation in enumerate(simulation_runs):
        success_times = [result['time'] for result in simulation.simulation_results['all_runs'] if result['success']]
        
        if success_times:
            # Create density plot for each run
            sns.kdeplot(success_times, label=f'Simulation with Weather Severity {i+1}', fill=True, alpha=0.2)
    
    # Add ideal time reference line
    ideal_time = simulation_runs[0].ideal_delivery_time  # Assuming the same for all runs
    plt.axvline(ideal_time, color='green', linestyle='--', label=f'Ideal Time: {ideal_time} minutes')
    
    plt.title('Distribution of Successful Delivery Times Across Simulation Runs', fontsize=14)
    plt.xlabel('Delivery Time (minutes)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.annotate('Note: Only successful deliveries are shown',
                xy=(0.5, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8),
                ha='center', fontsize=11)
    
    plt.tight_layout()
    plt.show()
    
    # 3. Cumulative Distribution Function (CDF) of Delivery Times
    plt.figure(figsize=(12, 7))
    
    for i, simulation in enumerate(simulation_runs):
        results = simulation.simulation_results['all_runs']
        success_times = [result['time'] for result in results if result['success']]
        
        if success_times:
            sorted_times = np.sort(success_times)
            y_all_attempts = np.arange(1, len(sorted_times) + 1) / len(results)
            
            plt.plot(sorted_times, y_all_attempts, marker='.', linestyle='-', 
                     label=f'Simulation with Weather Severity {i+1} ({len(success_times)}/{len(results)} = {len(success_times)/len(results)*100:.1f}%)')
    
    # Add ideal time reference line
    plt.axvline(ideal_time, color='green', linestyle='--', label=f'Ideal Time: {ideal_time} minutes')
    
    plt.title('Cumulative Probability of Delivery Times Across Simulation Runs', fontsize=14)
    plt.xlabel('Delivery Time (minutes)', fontsize=12)
    plt.ylabel('Cumulative Probability', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
    # 4. Common reasons for failure across runs
    failure_data = df[~df['success']]
    
    if not failure_data.empty:
        # plt.figure(figsize=(12, 7))
        
        # Count failures by reason and run
        failure_counts = pd.crosstab(failure_data['reason'], failure_data['run'])
        
        # Plot stacked bar chart
        failure_counts.plot(kind='bar', stacked=False, figsize=(12, 7), colormap='viridis')
        
        plt.title('Failure Reasons Across Simulation Runs', fontsize=14)
        plt.xlabel('Failure Reason', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.legend(title='Simulation with Weather Severity')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()