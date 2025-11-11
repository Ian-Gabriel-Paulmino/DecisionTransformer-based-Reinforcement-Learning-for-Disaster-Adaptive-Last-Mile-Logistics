"""
Step 4: Evaluate trained model
"""
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from DecisionTransformer.inference.evaluate import compare_dt_vs_nna

if __name__ == '__main__':
    print("=" * 60)
    print("STEP 4: MODEL EVALUATION")
    print("=" * 60)

    # Compute absolute paths relative to project root
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(BASE_DIR, 'checkpoints', 'best_model.pt')
    encoders_path = os.path.join(BASE_DIR, 'data', 'processed', 'encoders.pkl')

    compare_dt_vs_nna(
        model_path=model_path,
        encoders_path=encoders_path,
        num_trials=10
    )
