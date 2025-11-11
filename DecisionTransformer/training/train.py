"""
Main training script
"""
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import pickle
from DecisionTransformer.model.decision_transformer import DecisionTransformer
from DecisionTransformer.model.config import DecisionTransformerConfig
from DecisionTransformer.model.config import create_optimized_config_for_data_size
from DecisionTransformer.data.data_preparation import prepare_data
from DecisionTransformer.training.trainer import DecisionTransformerTrainer

# Import your NetworkMap utility
from DecisionTransformer.Utilities import NetworkMap


def main():
    """
    Improved training pipeline
    """
    print("="*60)
    print("IMPROVED Decision Transformer Training")
    print("="*60)
    
    # ========== Load Network ==========
    print("\n1. Loading network...")
    place_query = "La Trinidad, Benguet, Philippines"
    network_map = NetworkMap(place_query)
    network_map.download_map()
    network_map.network_to_networkx()
    
    start_node, delivery_nodes = network_map.select_fixed_points(num_delivery_points=20)
    print(f"   Start node: {start_node}")
    print(f"   Number of delivery nodes: {len(delivery_nodes)}")
    
    # ========== Prepare Data ==========
    print("\n2. Preparing data...")
    train_loader, val_loader, test_loader, encoders = prepare_data(
        trajectory_file='data/processed/trajectories_all.pkl',
        graph=network_map.G,
        delivery_nodes=delivery_nodes,
        test_size=0.2,
        val_size=0.1,
        batch_size=32,  # Smaller batch for better gradient estimates
        num_workers=0
    )
    
    # Get dataset size for config
    num_train_samples = len(train_loader.dataset)
    print(f"   Training samples: {num_train_samples}")
    
    # ========== Create Optimized Model ==========
    print("\n3. Creating optimized model...")
    config = create_optimized_config_for_data_size(num_train_samples)
    config.num_actions = len(delivery_nodes)
    config.state_dim = 76
    
    model = DecisionTransformer(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Samples per parameter: {num_train_samples/total_params:.2f}")
    
    # Rule of thumb check
    if num_train_samples / total_params < 0.05:
        print("   ⚠ WARNING: Very few samples per parameter")
        print("   Consider collecting more data or using smaller model")
    elif num_train_samples / total_params < 0.1:
        print("   ⚠ CAUTION: Low samples per parameter")
        print("   Model may overfit - watch validation loss")
    else:
        print("   ✓ Good samples-to-parameters ratio")
    
    # ========== Create Improved Trainer ==========
    print("\n4. Creating improved trainer...")
    trainer = DecisionTransformerTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir='checkpoints'
    )
    
    # ========== Train ==========
    print("\n5. Starting improved training...")
    print("   Innovations enabled:")
    print("   ✓ Quality-weighted loss")
    print("   ✓ Return aspiration")
    print("   ✓ Optimized model size")
    print()
    
    trainer.train(num_epochs=150, save_every=10)  # More epochs for smaller model
    
    # ========== Save Encoders ==========
    print("\n6. Saving encoders...")
    with open('data/processed/encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    print("   ✓ Saved encoders to data/processed/encoders.pkl")
    
    print("\n" + "="*60)
    print("Training pipeline complete!")
    print("="*60)
    
    # ========== Training Summary ==========
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Data:")
    print(f"  Training samples: {num_train_samples}")
    print(f"  Model parameters: {total_params:,}")
    print(f"  Samples/param ratio: {num_train_samples/total_params:.2f}")
    print(f"\nModel:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Attention heads: {config.num_heads}")
    print(f"\nTraining:")
    print(f"  Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"\nNext steps:")
    print(f"  1. Check TensorBoard: tensorboard --logdir=runs/")
    print(f"  2. Evaluate: python scripts/04_evaluate_model.py")
    print(f"  3. If performance poor, collect more data and retrain")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
