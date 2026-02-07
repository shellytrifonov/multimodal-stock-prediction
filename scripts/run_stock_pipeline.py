import subprocess
import sys
import os

print("=" * 70)
print("Stock Data Processing & Training Pipeline")
print("=" * 70)

steps = [
    # "python data/fetch_stock_data.py",  # Skip if data already in DB
    "python pipelines/stock/preprocess_stock.py",
    "python pipelines/stock/build_stock_training_data.py",
    "python models/train_stock_lstm.py"
]

for i, cmd in enumerate(steps, 1):
    print(f"\n[{i}/{len(steps)}] Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\nError at step {i}. Exiting.")
        sys.exit(1)

print("\n" + "=" * 70)
print("Stock Pipeline Complete!")
print("=" * 70)
