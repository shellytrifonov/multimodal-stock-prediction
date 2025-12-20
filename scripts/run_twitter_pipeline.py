import subprocess
import sys
import os

print("=" * 70)
print("Twitter Sentiment Analysis Pipeline")
print("=" * 70)

steps = [
    "python data/fetch_twitter_data.py",
    "python pipelines/twitter/preprocess_tweets.py",
    "python pipelines/twitter/twitter_sentiment.py",
    "python pipelines/twitter/aggregate_hourly_twitter.py",
    "python pipelines/twitter/build_twitter_training_data.py",
    "python models/train_twitter_lstm.py"
]

for i, cmd in enumerate(steps, 1):
    print(f"\n[{i}/{len(steps)}] Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\nError at step {i}. Exiting.")
        sys.exit(1)

print("\n" + "=" * 70)
print("Twitter Pipeline Complete!")
print("=" * 70)
