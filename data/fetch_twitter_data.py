import sys
import os

# Modify Python path to enable imports from parent directory, regardless of execution context
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kagglehub
import pandas as pd
import pymongo
from pymongo import ASCENDING, DESCENDING
from datetime import datetime

# Import database configuration parameters from central configuration module
from config.db_config import (
    MONGO_URI,
    DB_NAME,
    COLLECTION_RAW_TWEETS,
    LIMIT_TICKERS,
)

# Configuration constants
COLLECTION_NAME = COLLECTION_RAW_TWEETS  # Target collection name from configuration
DATASET_NAME = "omermetinn/tweets-about-the-top-companies-from-2015-to-2020"


def init_db_and_indexes(db):
    """
    Initialize database indexes for query optimization.
    
    Creates indexes to improve retrieval performance:
    - Ticker symbol index: Enables efficient filtering by stock ticker
    - Timestamp index: Supports fast chronological sorting and range queries
    - Unique post_id index: Prevents duplicate tweet documents
    """
    print("Checking and creating indexes...")
    tweets_collection = db[COLLECTION_NAME]

    # Create index for efficient retrieval by ticker symbol
    tweets_collection.create_index([("ticker", ASCENDING)])

    # Create index for chronological sorting and temporal queries
    tweets_collection.create_index([("created_at", DESCENDING)])

    # Unique index to avoid inserting the same tweet multiple times
    tweets_collection.create_index([("post_id", ASCENDING)], unique=True)

    print("Indexes created successfully.")


def main():
    # Step 1: Download dataset from Kaggle via KaggleHub API
    print(f"Downloading dataset '{DATASET_NAME}' from Kaggle...")
    try:
        path = kagglehub.dataset_download(DATASET_NAME)
        print(f"Dataset downloaded to: {path}")
    except Exception as e:
        print(f"ERROR: Failed to download dataset: {e}")
        return

    # Construct absolute file paths for dataset CSVs
    tweet_file = os.path.join(path, "Tweet.csv")
    company_file = os.path.join(path, "Company_Tweet.csv")

    # Step 2: Establish database connection and initialize schema
    try:
        client = pymongo.MongoClient(MONGO_URI)
        db = client[DB_NAME]
        init_db_and_indexes(db)
        collection = db[COLLECTION_NAME]
        print(f"Connected to MongoDB at {MONGO_URI}")
    except Exception as e:
        print(f"ERROR: Failed to connect to MongoDB: {e}")
        return

    # Optional: clear existing documents before ingestion
    try:
        choice = input(
            f"Clear existing documents in collection '{COLLECTION_NAME}' before ingestion? [y/N]: "
        ).strip().lower()
        if choice == "y":
            result = collection.delete_many({})
            print(
                f"Cleared {result.deleted_count} existing documents from '{COLLECTION_NAME}'."
            )
    except Exception as e:
        print(f"WARNING: Failed to clear collection: {e}")

    # Step 3: Load tweet-to-ticker mapping into memory for efficient lookups
    print("Loading ticker mapping into memory...")
    if not os.path.exists(company_file):
        print(f"ERROR: File not found: {company_file}")
        return

    try:
        # Read CSV with string type enforcement to preserve tweet ID integrity
        ticker_df = pd.read_csv(company_file, dtype={"tweet_id": str})

        # If LIMIT_TICKERS is defined, keep only those tickers
        if LIMIT_TICKERS:
            ticker_df = ticker_df[
                ticker_df["ticker_symbol"].isin(LIMIT_TICKERS)
            ]

        # Build dictionary for O(1) lookup complexity during ingestion
        tweet_to_ticker = dict(zip(ticker_df.tweet_id, ticker_df.ticker_symbol))
        print(f"Loaded {len(tweet_to_ticker)} ticker associations.")
    except Exception as e:
        print(f"ERROR: Failed to process Company_Tweet.csv: {e}")
        return

    # Step 4: Process and ingest tweet data with ticker associations
    print("Starting ingestion of tweets (Tweet.csv)...")

    if not os.path.exists(tweet_file):
        print(f"ERROR: File not found: {tweet_file}")
        return

    chunk_size = 10000
    total_inserted = 0

    # Process CSV in chunks to prevent memory overflow on large datasets
    for chunk in pd.read_csv(
        tweet_file, chunksize=chunk_size, dtype={"tweet_id": str}
    ):
        documents = []

        for index, row in chunk.iterrows():
            t_id = str(row["tweet_id"])

            # Filter tweets with valid ticker associations (ignore unassociated tweets)
            if t_id in tweet_to_ticker:
                associated_ticker = tweet_to_ticker[t_id]
                try:
                    # Convert Unix timestamp (epoch time) to Python datetime object for temporal analysis
                    post_date = datetime.fromtimestamp(int(row["post_date"]))

                    # Structure document according to database schema
                    doc = {
                        "ticker": associated_ticker,
                        "platform": "Twitter",
                        "post_id": t_id,
                        "text": row["body"],
                        "author": row["writer"],
                        "created_at": post_date,
                        "likes_count": int(row["like_num"]),
                        "retweet_count": int(row["retweet_num"]),
                        "comment_count": int(row["comment_num"]),
                        "sentiment_score": None,  # Reserved for sentiment analysis pipeline stage
                    }
                    documents.append(doc)
                except Exception:
                    # Skip rows with malformed data to maintain data quality
                    continue

        # Perform bulk insertion for improved efficiency
        if documents:
            try:
                # Use unordered insert to continue processing despite individual document failures (e.g., duplicate keys)
                collection.insert_many(documents, ordered=False)
                total_inserted += len(documents)
                print(f"Inserted {len(documents)} tweets... (Total: {total_inserted})")
            except Exception:
                # Ignore insertion errors (e.g., duplicates) and continue
                pass

    print(f"\nIngestion complete. Total tweets ingested: {total_inserted}")


if __name__ == "__main__":
    main()