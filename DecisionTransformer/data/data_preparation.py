"""
Main data preparation pipeline

This module orchestrates the entire data preparation process:
1. Load raw trajectories
2. Create encoders
3. Split into train/val/test
4. Create PyTorch DataLoaders
"""
import pickle
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from .state_encoder import StateEncoder
from .action_encoder import ActionEncoder
from .trajectory_dataset import DeliveryTrajectoryDataset, collate_trajectories


def prepare_data(trajectory_file, graph, delivery_nodes, test_size=0.2, val_size=0.1, 
                 batch_size=64, num_workers=0):
    """
    Load and prepare trajectory data for training
    
    This function:
    1. Loads raw trajectories from pickle file
    2. Creates state and action encoders
    3. Splits data into train/validation/test sets
    4. Creates PyTorch DataLoaders for each split
    
    Args:
        trajectory_file: Path to pickled trajectory data
        graph: NetworkX graph of road network
        delivery_nodes: List of delivery node IDs
        test_size: Fraction of data for test set (default: 0.2)
        val_size: Fraction of remaining data for validation (default: 0.1)
        batch_size: Batch size for DataLoaders (default: 64)
        num_workers: Number of workers for data loading (default: 0)
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, encoders)
            - train_loader: DataLoader for training
            - val_loader: DataLoader for validation
            - test_loader: DataLoader for testing
            - encoders: Dict with 'state_encoder' and 'action_encoder'
    """
    # Load trajectories
    print(f"Loading trajectories from {trajectory_file}...")
    with open(trajectory_file, 'rb') as f:
        all_trajectories = pickle.load(f)
    
    print(f"Loaded {len(all_trajectories)} trajectories")
    
    # Create encoders
    print("Creating encoders...")
    state_encoder = StateEncoder(graph, node_embedding_dim=64)
    action_encoder = ActionEncoder(delivery_nodes)
    
    print(f"State dimension: {state_encoder.state_dim}")
    print(f"Number of actions: {action_encoder.num_actions}")
    
    # Split data: first split off test, then split remaining into train/val
    train_val, test = train_test_split(
        all_trajectories, 
        test_size=test_size, 
        random_state=42
    )
    
    train, val = train_test_split(
        train_val, 
        test_size=val_size/(1-test_size),  # Adjust val_size for remaining data
        random_state=42
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(train)} trajectories")
    print(f"  Val:   {len(val)} trajectories")
    print(f"  Test:  {len(test)} trajectories")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = DeliveryTrajectoryDataset(train, state_encoder, action_encoder)
    val_dataset = DeliveryTrajectoryDataset(val, state_encoder, action_encoder)
    test_dataset = DeliveryTrajectoryDataset(test, state_encoder, action_encoder)
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_trajectories, 
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_trajectories, 
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_trajectories, 
        num_workers=num_workers
    )
    
    # Package encoders
    encoders = {
        'state_encoder': state_encoder,
        'action_encoder': action_encoder
    }
    
    print("\n✓ Data preparation complete!")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    
    return train_loader, val_loader, test_loader, encoders