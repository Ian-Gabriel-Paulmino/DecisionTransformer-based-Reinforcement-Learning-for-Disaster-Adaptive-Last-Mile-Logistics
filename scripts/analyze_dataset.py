import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

class TrajectoryAnalyzer:
    """
    Analyze trajectory data for Decision Transformer training.
    Computes return-to-go and various statistics for disaster-adaptive routing.
    """
    
    def __init__(self, pickle_path: str = "data/trajectories_all.pkl"):
        self.pickle_path = pickle_path
        self.trajectories = None
        self.df_steps = None
        self.df_trajectories = None
        
    def load_trajectories(self):
        """Load trajectories from pickle file."""
        print(f"Loading trajectories from {self.pickle_path}...")
        with open(self.pickle_path, 'rb') as f:
            self.trajectories = pickle.load(f)
        print(f"Loaded {len(self.trajectories)} trajectories")
        return self.trajectories
    
    def compute_return_to_go(self, rewards: List[float]) -> List[float]:
        """
        Compute return-to-go for a trajectory.
        Return-to-go at step t = sum of all future rewards from step t onwards.
        
        For routing: since rewards are negative (time costs), 
        return-to-go represents remaining cost to complete route.
        """
        rtg = []
        cumulative = 0
        # Go backwards through rewards
        for reward in reversed(rewards):
            cumulative += reward
            rtg.append(cumulative)
        # Reverse to get forward order
        rtg.reverse()
        return rtg
    
    def trajectory_to_dataframe(self, trajectory: Dict, traj_id: int) -> pd.DataFrame:
        """
        Convert a single trajectory to a dataframe with one row per step.
        Includes return-to-go computation.
        Handles missing fields gracefully.
        """
        steps = trajectory['trajectory']
        
        # Extract rewards first to compute return-to-go
        rewards = [step['reward'] for step in steps]
        returns_to_go = self.compute_return_to_go(rewards)
        
        rows = []
        for step_idx, step in enumerate(steps):
            state = step.get('state', {})
            next_state = step.get('next_state', {})
            disaster_context = state.get('disaster_context', {})
            next_disaster_context = next_state.get('disaster_context', {})
            disasters_encountered = step.get('disasters_encountered', {})
            
            row = {
                # Trajectory identifiers
                'trajectory_id': traj_id,
                'step': step_idx,
                'position_in_route': state.get('position_in_route', step_idx),
                
                # State information
                'current_node': state.get('current_node', None),
                'num_remaining_nodes': len(state.get('remaining_nodes', [])),
                'rain_intensity': state.get('rain_intensity', None),
                'floods_nearby': disaster_context.get('floods_nearby', 0),
                'landslides_nearby': disaster_context.get('landslides_nearby', 0),
                
                # Action and outcome
                'action': step.get('action', None),
                'reward': step.get('reward', 0),
                'return_to_go': returns_to_go[step_idx],
                
                # Step metrics
                'step_time': step.get('step_time', 0),
                'step_distance': step.get('step_distance', 0),
                'floods_encountered': disasters_encountered.get('floods', 0),
                'landslides_encountered': disasters_encountered.get('landslides', 0),
                'blocked': step.get('blocked', False),
                
                # Next state info (useful for transition analysis)
                'next_floods_nearby': next_disaster_context.get('floods_nearby', 0),
                'next_landslides_nearby': next_disaster_context.get('landslides_nearby', 0),
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def create_step_dataframe(self) -> pd.DataFrame:
        """
        Create a dataframe with one row per step across all trajectories.
        This is the main format for step-level analysis.
        """
        if self.trajectories is None:
            self.load_trajectories()
        
        print("Creating step-level dataframe...")
        all_steps = []
        
        for traj_id, traj in enumerate(self.trajectories):
            traj_df = self.trajectory_to_dataframe(traj, traj_id)
            all_steps.append(traj_df)
        
        self.df_steps = pd.concat(all_steps, ignore_index=True)
        print(f"Created dataframe with {len(self.df_steps)} total steps")
        return self.df_steps
    
    def create_trajectory_summary(self) -> pd.DataFrame:
        """
        Create a summary dataframe with one row per trajectory.
        Useful for comparing overall trajectory performance.
        """
        if self.df_steps is None:
            self.create_step_dataframe()
        
        print("Creating trajectory summary...")
        
        # Group by trajectory and compute summary statistics
        summary = self.df_steps.groupby('trajectory_id').agg({
            'reward': ['sum', 'mean', 'std', 'min', 'max'],
            'step_time': ['sum', 'mean', 'std'],
            'step_distance': ['sum', 'mean'],
            'floods_encountered': 'sum',
            'landslides_encountered': 'sum',
            'blocked': 'sum',
            'step': 'count',  # Number of steps
            'rain_intensity': 'first',  # Assuming constant per trajectory
        }).reset_index()
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
        
        # Rename for clarity
        summary.rename(columns={
            'reward_sum': 'total_reward',
            'reward_mean': 'avg_reward',
            'reward_std': 'reward_std',
            'reward_min': 'worst_step_reward',
            'reward_max': 'best_step_reward',
            'step_time_sum': 'total_time',
            'step_time_mean': 'avg_step_time',
            'step_time_std': 'step_time_std',
            'step_distance_sum': 'total_distance',
            'step_distance_mean': 'avg_step_distance',
            'floods_encountered_sum': 'total_floods',
            'landslides_encountered_sum': 'total_landslides',
            'blocked_sum': 'total_blocked_steps',
            'step_count': 'num_steps',
            'rain_intensity_first': 'rain_intensity'
        }, inplace=True)
        
        # Add initial return-to-go (total expected cost at start)
        initial_rtg = self.df_steps[self.df_steps['step'] == 0].set_index('trajectory_id')['return_to_go']
        summary['initial_return_to_go'] = summary['trajectory_id'].map(initial_rtg)
        
        self.df_trajectories = summary
        print(f"Created summary for {len(self.df_trajectories)} trajectories")
        return self.df_trajectories
    
    def get_quality_metrics(self, percentile: float = 75) -> Dict[str, float]:
        """
        Get quality thresholds for filtering high-quality trajectories.
        Uses percentile-based filtering (lower time = better).
        """
        if self.df_trajectories is None:
            self.create_trajectory_summary()
        
        threshold_time = np.percentile(self.df_trajectories['total_time'], percentile)
        threshold_reward = np.percentile(self.df_trajectories['total_reward'], 100 - percentile)
        
        metrics = {
            'time_threshold': threshold_time,
            'reward_threshold': threshold_reward,
            'percentile': percentile
        }
        
        print(f"\nQuality Metrics (top {100-percentile}% performers):")
        print(f"  Max total time: {threshold_time:.2f}")
        print(f"  Min total reward: {threshold_reward:.2f}")
        
        return metrics
    
    def filter_quality_trajectories(self, percentile: float = 75) -> pd.DataFrame:
        """
        Filter for high-quality trajectories (top performers).
        """
        if self.df_trajectories is None:
            self.create_trajectory_summary()
        
        metrics = self.get_quality_metrics(percentile)
        
        quality_mask = (
            (self.df_trajectories['total_time'] <= metrics['time_threshold']) &
            (self.df_trajectories['total_reward'] >= metrics['reward_threshold'])
        )
        
        quality_trajs = self.df_trajectories[quality_mask]
        quality_ids = set(quality_trajs['trajectory_id'])
        
        print(f"\nFiltered {len(quality_trajs)} quality trajectories ({len(quality_trajs)/len(self.df_trajectories)*100:.1f}%)")
        
        # Also filter step dataframe
        quality_steps = self.df_steps[self.df_steps['trajectory_id'].isin(quality_ids)]
        
        return quality_trajs, quality_steps
    
    def print_basic_stats(self):
        """Print basic statistics about the dataset."""
        if self.df_trajectories is None:
            self.create_trajectory_summary()
        
        print("\n" + "="*60)
        print("TRAJECTORY DATASET STATISTICS")
        print("="*60)
        
        print(f"\nTotal Trajectories: {len(self.df_trajectories)}")
        print(f"Total Steps: {len(self.df_steps)}")
        print(f"Avg Steps per Trajectory: {self.df_trajectories['num_steps'].mean():.1f}")
        
        print("\n--- Performance Metrics ---")
        print(f"Total Time (mean ± std): {self.df_trajectories['total_time'].mean():.2f} ± {self.df_trajectories['total_time'].std():.2f}")
        print(f"Total Time (min/max): {self.df_trajectories['total_time'].min():.2f} / {self.df_trajectories['total_time'].max():.2f}")
        print(f"Total Reward (mean ± std): {self.df_trajectories['total_reward'].mean():.2f} ± {self.df_trajectories['total_reward'].std():.2f}")
        
        print("\n--- Disaster Encounters ---")
        print(f"Avg Floods per Trajectory: {self.df_trajectories['total_floods'].mean():.2f}")
        print(f"Avg Landslides per Trajectory: {self.df_trajectories['total_landslides'].mean():.2f}")
        print(f"Trajectories with Blocked Steps: {(self.df_trajectories['total_blocked_steps'] > 0).sum()}")
        
        print("\n--- Weather Distribution ---")
        weather_dist = self.df_trajectories['rain_intensity'].value_counts().sort_index()
        for intensity, count in weather_dist.items():
            print(f"  Rain Intensity {intensity}: {count} trajectories ({count/len(self.df_trajectories)*100:.1f}%)")
        
        print("\n--- Return-to-Go Statistics ---")
        print(f"Initial RTG (mean ± std): {self.df_trajectories['initial_return_to_go'].mean():.2f} ± {self.df_trajectories['initial_return_to_go'].std():.2f}")
        print(f"Initial RTG range: [{self.df_trajectories['initial_return_to_go'].min():.2f}, {self.df_trajectories['initial_return_to_go'].max():.2f}]")
        
    def plot_analysis(self, save_dir: str = "analysis_plots"):
        """Generate visualization plots for trajectory analysis."""
        Path(save_dir).mkdir(exist_ok=True)
        
        if self.df_trajectories is None:
            self.create_trajectory_summary()
        
        # Set style
        sns.set_style("whitegrid")
        
        # 1. Performance Distribution
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0, 0].hist(self.df_trajectories['total_time'], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(self.df_trajectories['total_time'].median(), color='red', linestyle='--', label='Median')
        axes[0, 0].set_xlabel('Total Time')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Total Trajectory Time')
        axes[0, 0].legend()
        
        axes[0, 1].hist(self.df_trajectories['total_reward'], bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[0, 1].axvline(self.df_trajectories['total_reward'].median(), color='red', linestyle='--', label='Median')
        axes[0, 1].set_xlabel('Total Reward (negative time)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Total Trajectory Reward')
        axes[0, 1].legend()
        
        axes[1, 0].scatter(self.df_trajectories['total_distance'], self.df_trajectories['total_time'], alpha=0.5)
        axes[1, 0].set_xlabel('Total Distance (m)')
        axes[1, 0].set_ylabel('Total Time (min)')
        axes[1, 0].set_title('Distance vs Time')
        
        disaster_counts = self.df_trajectories['total_floods'] + self.df_trajectories['total_landslides']
        axes[1, 1].scatter(disaster_counts, self.df_trajectories['total_time'], alpha=0.5, color='red')
        axes[1, 1].set_xlabel('Total Disasters Encountered')
        axes[1, 1].set_ylabel('Total Time (min)')
        axes[1, 1].set_title('Disaster Impact on Performance')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/performance_distributions.png", dpi=300, bbox_inches='tight')
        print(f"Saved: {save_dir}/performance_distributions.png")
        plt.close()
        
        # 2. Return-to-Go Analysis
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Sample some trajectories to plot RTG curves
        sample_ids = np.random.choice(self.df_trajectories['trajectory_id'].values, 
                                     min(20, len(self.df_trajectories)), replace=False)
        
        for traj_id in sample_ids:
            traj_data = self.df_steps[self.df_steps['trajectory_id'] == traj_id]
            axes[0].plot(traj_data['step'], traj_data['return_to_go'], alpha=0.3)
        
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Return-to-Go')
        axes[0].set_title(f'Return-to-Go Curves (sample of {len(sample_ids)} trajectories)')
        axes[0].grid(True, alpha=0.3)
        
        # Initial RTG distribution
        axes[1].hist(self.df_trajectories['initial_return_to_go'], bins=50, edgecolor='black', alpha=0.7, color='green')
        axes[1].axvline(self.df_trajectories['initial_return_to_go'].median(), color='red', linestyle='--', label='Median')
        axes[1].set_xlabel('Initial Return-to-Go')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Initial Return-to-Go')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/return_to_go_analysis.png", dpi=300, bbox_inches='tight')
        print(f"Saved: {save_dir}/return_to_go_analysis.png")
        plt.close()
        
        # 3. Weather Impact Analysis
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        weather_performance = self.df_trajectories.groupby('rain_intensity')['total_time'].agg(['mean', 'std', 'count'])
        axes[0].bar(weather_performance.index, weather_performance['mean'], 
                   yerr=weather_performance['std'], capsize=5, alpha=0.7)
        axes[0].set_xlabel('Rain Intensity')
        axes[0].set_ylabel('Mean Total Time (min)')
        axes[0].set_title('Performance by Weather Severity')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Box plot
        self.df_trajectories.boxplot(column='total_time', by='rain_intensity', ax=axes[1])
        axes[1].set_xlabel('Rain Intensity')
        axes[1].set_ylabel('Total Time (min)')
        axes[1].set_title('Time Distribution by Weather')
        plt.suptitle('')  # Remove automatic title
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/weather_impact.png", dpi=300, bbox_inches='tight')
        print(f"Saved: {save_dir}/weather_impact.png")
        plt.close()
        
        print(f"\nAll plots saved to {save_dir}/")
    
    def save_dataframes(self, output_dir: str = "data"):
        """Save the processed dataframes for later use."""
        Path(output_dir).mkdir(exist_ok=True)
        
        if self.df_steps is None:
            self.create_step_dataframe()
        if self.df_trajectories is None:
            self.create_trajectory_summary()
        
        steps_path = f"{output_dir}/trajectory_steps.csv"
        summary_path = f"{output_dir}/trajectory_summary.csv"
        
        self.df_steps.to_csv(steps_path, index=False)
        self.df_trajectories.to_csv(summary_path, index=False)
        
        print(f"\nSaved dataframes:")
        print(f"  Steps: {steps_path}")
        print(f"  Summary: {summary_path}")


def main():
    """Main analysis pipeline."""
    
    # Initialize analyzer
    analyzer = TrajectoryAnalyzer("data/processed/trajectories_all.pkl")
    
    # Load and process data
    analyzer.load_trajectories()
    analyzer.create_step_dataframe()
    analyzer.create_trajectory_summary()
    
    # Print statistics
    analyzer.print_basic_stats()
    
    # Get quality filtering metrics
    print("\n" + "="*60)
    print("QUALITY FILTERING ANALYSIS")
    print("="*60)
    quality_trajs, quality_steps = analyzer.filter_quality_trajectories(percentile=75)
    
    print(f"\nQuality trajectory IDs: {sorted(quality_trajs['trajectory_id'].values)[:10]}... (showing first 10)")
    
    # Generate plots
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    analyzer.plot_analysis()
    
    # Save processed data
    print("\n" + "="*60)
    print("SAVING PROCESSED DATA")
    print("="*60)
    analyzer.save_dataframes()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nNext steps for Decision Transformer:")
    print("  1. Use quality-filtered trajectories for training")
    print("  2. Return-to-go values are pre-computed in the dataframes")
    print("  3. Consider using initial_return_to_go as target conditioning")
    print("  4. Weather severity distribution shows data balance")


if __name__ == "__main__":
    main()