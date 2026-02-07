import subprocess
import sys
import os

print("=" * 70)
print("Hybrid Fusion Model Training Pipeline")
print("=" * 70)

print("\nPrerequisites:")
print("  - Stock LSTM trained (models/trained/stock_lstm_trained.pth)")
print("  - Twitter LSTM trained (models/trained/twitter_lstm_trained.pth)")

print("\nChecking for pre-trained models...")
required_models = [
    "models/trained/stock_lstm_trained.pth",
    "models/trained/twitter_lstm_trained.pth"
]

missing_models = []
for model_path in required_models:
    if os.path.exists(model_path):
        print(f"  ✓ Found: {model_path}")
    else:
        print(f"  ✗ Missing: {model_path}")
        missing_models.append(model_path)

if missing_models:
    print("\n" + "=" * 70)
    print("ERROR: Missing pre-trained models!")
    print("=" * 70)
    print("\nPlease train individual models first:")
    print("  python scripts/run_stock_pipeline.py")
    print("  python scripts/run_twitter_pipeline.py")
    sys.exit(1)

print("\n" + "=" * 70)

steps = [
    "python -m models.train_hybrid_fusion"
]

for i, cmd in enumerate(steps, 1):
    print(f"\n[{i}/{len(steps)}] Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\nError at step {i}. Exiting.")
        sys.exit(1)

print("\n" + "=" * 70)
print("Hybrid Fusion Pipeline Complete!")
print("=" * 70)
print("\nFinal model saved: models/trained/hybrid_fusion_trained.pth")
