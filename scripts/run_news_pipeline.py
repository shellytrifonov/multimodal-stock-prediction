import subprocess
import sys
import os

print("=" * 70)
print("News Sentiment Analysis Pipeline")
print("=" * 70)

steps = [
    "python data/fetch_news_data.py",
    "python pipelines/news/preprocess_news.py",
    "python pipelines/news/news_sentiment.py",
    "python pipelines/news/aggregate_hourly_news.py",
    "python pipelines/news/build_news_training_data.py",
    "python models/train_news_lstm.py"
]

for i, cmd in enumerate(steps, 1):
    print(f"\n[{i}/{len(steps)}] Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\nError at step {i}. Exiting.")
        sys.exit(1)

print("\n" + "=" * 70)
print("News Pipeline Complete!")
print("=" * 70)
