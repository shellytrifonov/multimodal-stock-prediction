import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pymongo
import pandas as pd
import numpy as np
from datetime import timedelta
# No scaling here - done after split to prevent leakage
from config.db_config import (
    MONGO_URI, DB_NAME,
    COLLECTION_STOCK_PRICES, LIMIT_TICKERS,
    COLLECTION_TWEETS_WITH_SENTIMENT
)

print("=" * 70)
print("Stock Data Preprocessing & Normalization")
print("=" * 70)


def detect_global_date_range(db):
    """
    Get date range from Twitter data + 90-day buffer for LSTM lookback.
    
    Returns:
        (start_date, end_date) as pandas Timestamps
    """
    print("\n1. Detecting date range from sentiment data...")
    
    twitter_collection = db[COLLECTION_TWEETS_WITH_SENTIMENT]
    
    min_dates = []
    max_dates = []
    
    # Get Twitter date range
    twitter_count = twitter_collection.count_documents({})
    if twitter_count > 0:
        twitter_dates = twitter_collection.find({}, {'created_at': 1}).sort('created_at', 1)
        twitter_min = next(twitter_dates, None)
        twitter_dates_max = twitter_collection.find({}, {'created_at': 1}).sort('created_at', -1)
        twitter_max = next(twitter_dates_max, None)
        
        if twitter_min and twitter_max:
            twitter_min_date = pd.to_datetime(twitter_min['created_at'])
            twitter_max_date = pd.to_datetime(twitter_max['created_at'])
            min_dates.append(twitter_min_date)
            max_dates.append(twitter_max_date)
            print(f"   Twitter: {twitter_min_date.date()} to {twitter_max_date.date()} ({twitter_count:,} tweets)")
    else:
        print("   No Twitter sentiment data found")
    
    if not min_dates:
        print("   ⚠ WARNING: No sentiment data, using default range")
        return pd.Timestamp('2015-01-01'), pd.Timestamp('2020-12-31')
    
    global_min = min(min_dates)
    global_max = max(max_dates)
    
    # 90-day buffer: 60 days for LSTM + 30-day margin
    buffer_days = 90
    stock_start_date = global_min - timedelta(days=buffer_days)
    stock_end_date = global_max
    
    print(f"\n   Sentiment range: {global_min.date()} to {global_max.date()}")
    print(f"   Stock range (with {buffer_days}-day buffer): {stock_start_date.date()} to {stock_end_date.date()}")
    
    return stock_start_date, stock_end_date


def preprocess_stock_data():
    """
    Clean and sort stock data (no scaling to prevent leakage).
    
    Pipeline: Load → Filter tickers → Sort → Handle missing values → Save
    
    Note: Scaling done after train/test split in build_stock_training_data.py
    """
    print("\n1. Connecting to MongoDB...")
    client = pymongo.MongoClient(MONGO_URI)
    db = client[DB_NAME]
    raw_collection = db[COLLECTION_STOCK_PRICES]
    cleaned_collection = db['cleaned_stock_prices']
    
    # Detect global date range from sentiment data
    stock_start_date, stock_end_date = detect_global_date_range(db)
    
    print("\n2. Clearing previous cleaned stock data...")
    cleaned_collection.delete_many({})
    print("   Ready for preprocessing.")
    
    print("\n3. Discovering tickers from database...")
    if LIMIT_TICKERS:
        tickers = LIMIT_TICKERS
        print(f"   Using specified tickers: {tickers}")
    else:
        tickers = raw_collection.distinct('ticker')
        print(f"   Found {len(tickers)} tickers: {tickers}")
    
    if not tickers:
        print("   ERROR: No tickers found. Run data ingestion first.")
        client.close()
        return
    
    print(f"\n4. Preprocessing stock data...")
    print("=" * 70)
    print(f"   Date filter: {stock_start_date.date()} to {stock_end_date.date()}")
    
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'adj_close']
    total_processed = 0
    total_cleaned = 0
    
    for ticker in tickers:
        print(f"\n   Processing {ticker}...")
        
        query = {'ticker': ticker}
        stock_docs = list(raw_collection.find(query))
        
        if not stock_docs:
            print(f"     No stock data found for {ticker}")
            continue
        
        print(f"     Found {len(stock_docs)} raw records")
        
        # Convert to DataFrame
        df = pd.DataFrame(stock_docs)
        df['date'] = pd.to_datetime(df['date'])
        
        # Ensure both sides are timezone-aware (UTC) to avoid tz-naive vs tz-aware comparisons
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")

        stock_start_date = pd.to_datetime(stock_start_date, utc=True)
        stock_end_date   = pd.to_datetime(stock_end_date,   utc=True)


        # Filter by detected date range
        df = df[(df['date'] >= stock_start_date) & (df['date'] <= stock_end_date)]
        
        if len(df) == 0:
            print(f"     No data in required date range, skipping {ticker}")
            continue
        
        # Sort by date ascending
        df = df.sort_values('date').reset_index(drop=True)
        
        # Check for required columns
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            print(f"     WARNING: Missing columns {missing_cols}, skipping {ticker}")
            continue
        
        # Handle missing values
        initial_nulls = df[feature_cols].isnull().sum().sum()
        if initial_nulls > 0:
            print(f"     Found {initial_nulls} missing values, applying forward fill...")
            df[feature_cols] = df[feature_cols].fillna(method='ffill')
            
            # If still NaN at the beginning, use backward fill
            remaining_nulls = df[feature_cols].isnull().sum().sum()
            if remaining_nulls > 0:
                df[feature_cols] = df[feature_cols].fillna(method='bfill')
                print(f"     Applied backward fill for initial NaNs")
        
        # Save RAW data (NO SCALING to prevent leakage)
        print(f"     Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"     Trading days: {len(df)}")
        print(f"     Saving RAW values (scaling will be done after train/test split)")
        
        # Prepare documents for MongoDB - store RAW values only
        cleaned_docs = []
        for idx, row in df.iterrows():
            doc = {
                'ticker': ticker,
                'date': row['date'],
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']),
                'adj_close': float(row['adj_close'])
            }
            cleaned_docs.append(doc)
        
        # Insert cleaned data
        if cleaned_docs:
            cleaned_collection.insert_many(cleaned_docs)
            total_cleaned += len(cleaned_docs)
            print(f"     ✓ Saved {len(cleaned_docs)} cleaned records")
        
        total_processed += len(stock_docs)
    
    print(f"\n5. Summary:")
    print("=" * 70)
    print(f"   Tickers processed: {len(tickers)}")
    print(f"   Total raw records: {total_processed}")
    print(f"   Total cleaned records: {total_cleaned}")
    print(f"   Features saved: {feature_cols}")
    print(f"   Scaling: NONE (will be applied after train/test split)")
    print(f"\n   Preprocessing complete - RAW values saved.")
    
    # Display sample statistics
    print(f"\n6. Sample Statistics:")
    print("=" * 70)
    sample_ticker = tickers[0]
    sample_docs = list(cleaned_collection.find({'ticker': sample_ticker}).limit(5))
    
    if sample_docs:
        print(f"   First 5 records for {sample_ticker}:")
        sample_df = pd.DataFrame(sample_docs)
        print(f"\n{sample_df[['date', 'open', 'high', 'low', 'close', 'volume']].to_string(index=False)}")
    
    client.close()


if __name__ == "__main__":
    preprocess_stock_data()
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
