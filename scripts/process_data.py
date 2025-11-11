import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from DecisionTransformer.data.data_preparation import prepare_data
from DecisionTransformer.Utilities import NetworkMap
import pickle


def main():
    print("=" * 60)
    print("STEP 2: DATA PROCESSING")
    print("=" * 60)

    print("\nProcessing trajectory data...")

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    processed_dir = os.path.join(BASE_DIR, 'data', 'processed')
    trajectory_path = os.path.join(processed_dir, 'trajectories_all.pkl')
    encoder_path = os.path.join(processed_dir, 'encoders.pkl')

    # Load network
    network_map = NetworkMap("La Trinidad, Benguet, Philippines")
    network_map.download_map()
    network_map.network_to_networkx()
    _, delivery_nodes = network_map.select_fixed_points(num_delivery_points=20)

    # Prepare data
    train_loader, val_loader, test_loader, encoders = prepare_data(
        trajectory_file=trajectory_path,
        graph=network_map.G,
        delivery_nodes=delivery_nodes,
        test_size=0.2,
        val_size=0.1
    )

    # Save encoders
    with open(encoder_path, 'wb') as f:
        pickle.dump(encoders, f)

    print("\n" + "=" * 60)
    print("✓ Data processing complete!")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Encoders saved to: {encoder_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
