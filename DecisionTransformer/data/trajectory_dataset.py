"""
PyTorch Dataset for trajectory data

This module defines the dataset class that feeds trajectory data to the 
Decision Transformer during training.
"""
import torch
from torch.utils.data import Dataset
import numpy as np


class DeliveryTrajectoryDataset(Dataset):
    """
    PyTorch Dataset for delivery trajectories
    
    Each trajectory is a sequence of (state, action, reward) tuples collected
    from the Monte Carlo simulation.
    """
    
    def __init__(self, trajectories, state_encoder, action_encoder, max_length=None):
        """
        Initialize dataset
        
        Args:
            trajectories: List of trajectory dictionaries
            state_encoder: StateEncoder instance
            action_encoder: ActionEncoder instance
            max_length: Optional maximum sequence length (for truncation)
        """
        self.trajectories = trajectories
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder
        self.max_length = max_length
        
        # Filter out empty trajectories
        self.valid_trajectories = [
            t for t in trajectories 
            if len(t['trajectory']) > 0
        ]
        
        print(f"Dataset created with {len(self.valid_trajectories)} valid trajectories")
    
    def __len__(self):
        """Return number of trajectories in dataset"""
        return len(self.valid_trajectories)
    
    def __getitem__(self, idx):
        """
        Get a single trajectory
        
        Args:
            idx: Index of trajectory
        
        Returns:
            dict with keys:
                - returns: Returns-to-go sequence (seq_length, 1)
                - states: State vectors (seq_length, state_dim)
                - actions: Action indices (seq_length,)
                - timesteps: Timestep indices (seq_length,)
                - seq_length: Length of sequence
                - success: Whether delivery was successful
                - rain_intensity: Weather condition
        """
        trajectory_data = self.valid_trajectories[idx]
        trajectory = trajectory_data['trajectory']
        
        # Extract sequences
        states = []
        actions = []
        rewards = []
        
        for step in trajectory:
            # Encode state
            state_vector = self.state_encoder.encode_state(step['state'])
            states.append(state_vector)
            
            # Encode action
            action_idx = self.action_encoder.encode_action(step['action'])
            actions.append(action_idx)
            
            # Get reward
            rewards.append(step['reward'])
        
        # Compute returns-to-go
        returns_to_go = self._compute_returns_to_go(rewards)
        
        # Convert to tensors
        returns_tensor = torch.tensor(returns_to_go, dtype=torch.float32).unsqueeze(-1)
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        timesteps_tensor = torch.arange(len(states), dtype=torch.long)
        
        # Truncate if needed
        if self.max_length and len(states) > self.max_length:
            returns_tensor = returns_tensor[:self.max_length]
            states_tensor = states_tensor[:self.max_length]
            actions_tensor = actions_tensor[:self.max_length]
            timesteps_tensor = timesteps_tensor[:self.max_length]
        
        return {
            'returns': returns_tensor,
            'states': states_tensor,
            'actions': actions_tensor,
            'timesteps': timesteps_tensor,
            'seq_length': len(states),
            'success': trajectory_data['success'],
            'rain_intensity': trajectory_data['rain_intensity']
        }
    
    def _compute_returns_to_go(self, rewards):
        """
        Compute return-to-go for each timestep
        
        Return-to-go at time t = sum of all future rewards from t onward
        
        Args:
            rewards: List of rewards at each timestep
        
        Returns:
            List of returns-to-go
        """
        returns = []
        running_return = 0
        
        # Iterate backwards through rewards
        for reward in reversed(rewards):
            running_return += reward
            returns.insert(0, running_return)
        
        return returns


def collate_trajectories(batch):
    """
    Collate function to pad sequences to same length
    
    PyTorch DataLoader requires all sequences in a batch to have the same length.
    This function pads shorter sequences with zeros.
    
    Args:
        batch: List of trajectory dictionaries
    
    Returns:
        dict with padded tensors:
            - returns: (batch_size, max_length, 1)
            - states: (batch_size, max_length, state_dim)
            - actions: (batch_size, max_length)
            - timesteps: (batch_size, max_length)
            - attention_mask: (batch_size, max_length * 3)
    """
    # Find maximum sequence length in batch
    max_length = max(item['seq_length'] for item in batch)
    
    batch_returns = []
    batch_states = []
    batch_actions = []
    batch_timesteps = []
    batch_masks = []
    
    for item in batch:
        seq_len = item['seq_length']
        pad_len = max_length - seq_len
        
        # Pad sequences with zeros
        padded_returns = torch.cat([
            item['returns'],
            torch.zeros(pad_len, 1)
        ])
        
        padded_states = torch.cat([
            item['states'],
            torch.zeros(pad_len, item['states'].shape[1])
        ])
        
        padded_actions = torch.cat([
            item['actions'],
            torch.zeros(pad_len, dtype=torch.long)
        ])
        
        padded_timesteps = torch.cat([
            item['timesteps'],
            torch.zeros(pad_len, dtype=torch.long)
        ])
        
        # Create attention mask for interleaved sequence
        # [R, s, a, R, s, a, ...] → mask length is seq_len * 3
        # 1 = real token, 0 = padding
        mask = torch.cat([
            torch.ones(seq_len * 3),
            torch.zeros(pad_len * 3)
        ])
        
        batch_returns.append(padded_returns)
        batch_states.append(padded_states)
        batch_actions.append(padded_actions)
        batch_timesteps.append(padded_timesteps)
        batch_masks.append(mask)
    
    return {
        'returns': torch.stack(batch_returns),
        'states': torch.stack(batch_states),
        'actions': torch.stack(batch_actions),
        'timesteps': torch.stack(batch_timesteps),
        'attention_mask': torch.stack(batch_masks)
    }