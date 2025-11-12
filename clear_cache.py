"""
Clear Python Cache and Verify Fix

Run this before training to ensure new trainer.py is loaded
"""
import os
import shutil
import sys

print("="*60)
print("CACHE CLEARING AND VERIFICATION")
print("="*60)

# Your project root
PROJECT_ROOT = r"C:\Users\Acer Nitro\Documents\CSC FILES\4th Year First Semester\Intellegent Systems\Transformer-Based Last Mile Logistics"

print(f"\nProject root: {PROJECT_ROOT}")

# Paths to clear
cache_paths = [
    os.path.join(PROJECT_ROOT, "DecisionTransformer", "training", "__pycache__"),
    os.path.join(PROJECT_ROOT, "DecisionTransformer", "__pycache__"),
    os.path.join(PROJECT_ROOT, "__pycache__"),
]

print("\nClearing Python caches...")
for cache_path in cache_paths:
    if os.path.exists(cache_path):
        try:
            shutil.rmtree(cache_path)
            print(f"  ✓ Cleared: {cache_path}")
        except Exception as e:
            print(f"  ✗ Failed to clear {cache_path}: {e}")
    else:
        print(f"  - Not found: {cache_path}")

print("\n" + "="*60)
print("VERIFICATION")
print("="*60)

# Verify trainer.py has the fix
trainer_path = os.path.join(PROJECT_ROOT, "DecisionTransformer", "training", "trainer.py")

if os.path.exists(trainer_path):
    with open(trainer_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for the fixed version markers
    has_version_marker = "Version 2.0" in content
    has_device_check = "assert weights.device == self.device" in content
    has_ones_like = "torch.ones_like(final_returns)" in content
    has_detach = ".min().detach()" in content
    
    print(f"\nTrainer.py checks:")
    print(f"  Version marker: {'✓' if has_version_marker else '✗'}")
    print(f"  Device assertion: {'✓' if has_device_check else '✗'}")
    print(f"  torch.ones_like: {'✓' if has_ones_like else '✗'}")
    print(f"  .detach() usage: {'✓' if has_detach else '✗'}")
    
    if all([has_version_marker, has_device_check, has_ones_like, has_detach]):
        print(f"\n✓ trainer.py has ALL fixes applied!")
    else:
        print(f"\n✗ trainer.py is MISSING some fixes!")
        print(f"\nYou need to replace trainer.py with the fixed version:")
        print(f"  1. Download trainer.py from outputs folder")
        print(f"  2. Replace: {trainer_path}")
else:
    print(f"\n✗ trainer.py not found at: {trainer_path}")

print("\n" + "="*60)
print("NEXT STEPS")
print("="*60)
print("""
1. If checks failed, replace trainer.py:
   - Copy trainer.py from outputs folder
   - Paste to: DecisionTransformer/training/trainer.py

2. Close any Python processes/IDEs

3. Run training:
   python scripts/train_model.py

4. You should see:
   ✓ Loading FIXED trainer.py - Version 2.0
   [No device mismatch errors]
""")

print("="*60)