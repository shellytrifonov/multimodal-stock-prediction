import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pymongo
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config.db_config import MONGO_URI, DB_NAME, COLLECTION_HOURLY_SENTIMENT_TWEETS, COLLECTION_STOCK_PRICES


def load_sentiment_data_by_ticker(db, ticker):
    """Load hourly sentiment features for specified ticker."""
    hourly_collection = db[COLLECTION_HOURLY_SENTIMENT_TWEETS]
    hourly_docs = list(hourly_collection.find({'ticker': ticker}).sort('hour', 1))
    
    if not hourly_docs:
        return None
    
    df = pd.DataFrame(hourly_docs)
    df['hour'] = pd.to_datetime(df['hour'])
    df = df.set_index('hour').sort_index()
    
    feature_cols = [
        'p_negative', 'p_neutral', 'p_positive',
        'mean_sentiment', 'tweet_count', 'sentiment_std',
        'max_sentiment', 'min_sentiment'
    ]
    df = df[feature_cols]
    
    return df


def load_financial_data_by_ticker(db, ticker):
    """Load daily stock prices and compute binary direction labels."""
    stock_collection = db[COLLECTION_STOCK_PRICES]
    stock_docs = list(stock_collection.find({'ticker': ticker}).sort('date', 1))
    
    if not stock_docs:
        return None
    
    df = pd.DataFrame(stock_docs)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    df['Next_Close'] = df['close'].shift(-1)
    df['Label'] = (df['Next_Close'] > df['close']).astype(int)
    df = df[:-1].copy()
    df = df.rename(columns={'date': 'Date', 'close': 'Close'})
    df['ticker'] = ticker
    
    return df


def extract_72hour_window(sentiment_df, prediction_time):
    """
    Extract 72-hour sentiment feature window with delta (momentum) features.
    
    CRITICAL: Window must END BEFORE market open on prediction day to avoid data leakage.
    Window ends at previous market close (4 PM on D-1) when predicting day D movement.
    """
    # End at previous day's market close (16:00 the day before prediction day)
    # This ensures NO information from prediction day enters the features
    end_time = prediction_time - timedelta(hours=24)  # 4 PM yesterday
    start_time = end_time - timedelta(hours=71)       # 72 hours = 71-hour span + start hour
    # Use periods=72 to get exactly 72 hourly timestamps (inclusive range would give 73)
    hour_range = pd.date_range(start=start_time, periods=72, freq='h')
    
    if len(hour_range) != 72:
        return None
    
    feature_cols = sentiment_df.columns.tolist()
    num_features = len(feature_cols)
    
    window_data = []
    prev_mean_sent = None
    prev_tweet_count = None
    
    for hour in hour_range:
        if hour in sentiment_df.index:
            row_values = sentiment_df.loc[hour, feature_cols].values
            
            # Extract mean_sentiment and tweet_count for delta calculation
            mean_sent = sentiment_df.loc[hour, 'mean_sentiment']
            tweet_count = sentiment_df.loc[hour, 'tweet_count']
            
            # Calculate deltas (momentum)
            if prev_mean_sent is not None:
                delta_mean_sent = mean_sent - prev_mean_sent
                delta_tweet_count = tweet_count - prev_tweet_count
            else:
                delta_mean_sent = 0.0
                delta_tweet_count = 0.0
            
            # Append original features + delta features
            row_with_deltas = np.concatenate([row_values, [delta_mean_sent, delta_tweet_count]])
            window_data.append(row_with_deltas)
            
            prev_mean_sent = mean_sent
            prev_tweet_count = tweet_count
        else:
            # Missing hour: fill with zeros (including deltas)
            window_data.append(np.zeros(num_features + 2))
            prev_mean_sent = None
            prev_tweet_count = None
    
    return np.array(window_data, dtype=np.float32)


def build_training_samples(db, prediction_hour=16):
    """
    Construct LSTM training dataset by aligning sentiment windows with price labels.
    
    IMPORTANT: Twitter window ends at PREVIOUS day's market close to prevent data leakage.
    For a sample on date D:
    - Twitter window: [D-4 16:00, D-1 16:00] (72 hours ending at yesterday's close)
    - Stock window: 60 days ending on D-1
    - Label: Predicts close(D) > close(D-1)
    - Stored date: D-1 (the observation day)
    
    This ensures features contain ONLY information available before predicting D's movement.
    """
    print("\n1. Discovering tickers from database...")
    hourly_collection = db[COLLECTION_HOURLY_SENTIMENT_TWEETS]
    
    tickers = hourly_collection.distinct('ticker')
    print(f"   Found {len(tickers)} tickers: {tickers}")
    
    if not tickers:
        print("   ERROR: No tickers found in hourly_sentiment_tweets collection.")
        return None, None, None, None
    
    print(f"\n2. Building training samples...")
    print("=" * 70)
    print(f"   Reference time: {prediction_hour}:00 each trading day")
    print(f"   Sentiment window: 72 hours ENDING at previous day's close (no leakage)")
    print(f"   Features: [p_neg, p_neu, p_pos, mean_sent, count, std, max, min, delta_mean_sent, delta_count]")
    print(f"   Label: Predicts next day's movement based on historical data only")
    
    X_list = []
    y_list = []
    date_list = []
    ticker_list = []
    total_skipped = 0
    
    for ticker in tickers:
        print(f"\n   Processing {ticker}...")
        
        sentiment_df = load_sentiment_data_by_ticker(db, ticker)
        stock_df = load_financial_data_by_ticker(db, ticker)
        
        if sentiment_df is None:
            print(f"     No sentiment data for {ticker}")
            continue
        
        if stock_df is None:
            print(f"     No stock data for {ticker}")
            continue
        
        print(f"     Sentiment: {len(sentiment_df)} hours")
        print(f"     Stock: {len(stock_df)} trading days")
        
        num_features = len(sentiment_df.columns)
        skipped = 0
        
        for idx, row in stock_df.iterrows():
            trading_date = row['Date']
            label = row['Label']
            prediction_time = trading_date.replace(hour=prediction_hour, minute=0, second=0)
            window = extract_72hour_window(sentiment_df, prediction_time)
            
            if window is None:
                skipped += 1
                continue
            
            window_start = prediction_time - timedelta(hours=72)
            if window_start < sentiment_df.index.min():
                skipped += 1
                continue
            
            X_list.append(window)
            y_list.append(label)
            date_list.append(trading_date)
            ticker_list.append(ticker)
        
        print(f"     Samples created: {len(stock_df) - skipped}")
        print(f"     Samples skipped: {skipped}")
        total_skipped += skipped
    
    # Convert to numpy arrays with float32 for memory efficiency
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    dates = np.array(date_list)
    tickers = np.array(ticker_list)
    
    print(f"\n3. Implementing chronological 80/20 train/test split...")
    print("=" * 70)
    
    # Check if we have any samples
    if len(X) == 0:
        print(f"   ERROR: No samples created!")
        print(f"   Total samples skipped: {total_skipped}")
        return None
    
    # Sort by date to ensure chronological order
    sorted_indices = np.argsort(dates)
    X = X[sorted_indices]
    y = y[sorted_indices]
    dates = dates[sorted_indices]
    tickers = tickers[sorted_indices]
    
    # Chronological split: first 80% train, last 20% test
    total_samples = len(X)
    train_size = int(total_samples * 0.8)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    dates_train = dates[:train_size]
    tickers_train = tickers[:train_size]
    
    X_test = X[train_size:]
    y_test = y[train_size:]
    dates_test = dates[train_size:]
    tickers_test = tickers[train_size:]
    
    print(f"   Total samples: {total_samples}")
    print(f"   Train samples: {len(X_train)} (first 80%)")
    print(f"   Test samples: {len(X_test)} (last 20%)")
    print(f"   Train date range: {pd.Timestamp(dates_train[0]).date()} to {pd.Timestamp(dates_train[-1]).date()}")
    print(f"   Test date range: {pd.Timestamp(dates_test[0]).date()} to {pd.Timestamp(dates_test[-1]).date()}")
    
    print(f"\n4. Results:")
    print("=" * 70)
    print(f"   Tickers processed: {len(tickers)}")
    print(f"   Total samples skipped: {total_skipped}")
    print(f"\n   Train set:")
    print(f"     Shape: {X_train.shape} (samples, timesteps, features)")
    print(f"     Up: {np.sum(y_train)} ({100*np.mean(y_train):.1f}%)")
    print(f"     Down: {len(y_train) - np.sum(y_train)} ({100*(1-np.mean(y_train)):.1f}%)")
    print(f"\n   Test set:")
    print(f"     Shape: {X_test.shape} (samples, timesteps, features)")
    print(f"     Up: {np.sum(y_test)} ({100*np.mean(y_test):.1f}%)")
    print(f"     Down: {len(y_test) - np.sum(y_test)} ({100*(1-np.mean(y_test)):.1f}%)")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'dates_train': dates_train,
        'tickers_train': tickers_train,
        'X_test': X_test,
        'y_test': y_test,
        'dates_test': dates_test,
        'tickers_test': tickers_test,
        'num_features': X_train.shape[2]
    }


def save_training_data(data_dict, output_file):
    """Save training dataset with pre-split train/test to compressed NumPy archive."""
    np.savez_compressed(output_file,
                       X_train=data_dict['X_train'],
                       y_train=data_dict['y_train'],
                       dates_train=[d.strftime('%Y-%m-%d') for d in data_dict['dates_train']],
                       tickers_train=data_dict['tickers_train'],
                       X_test=data_dict['X_test'],
                       y_test=data_dict['y_test'],
                       dates_test=[d.strftime('%Y-%m-%d') for d in data_dict['dates_test']],
                       tickers_test=data_dict['tickers_test'],
                       num_features=data_dict['num_features'])
    
    file_size_mb = os.path.getsize(output_file) / 1024 / 1024
    print(f"\n5. Saved: {output_file}")
    print("=" * 70)
    print(f"   File size: {file_size_mb:.2f} MB")
    print("\n   Loading instructions:")
    print(f"     data = np.load('{output_file}')")
    print(f"     X_train, y_train = data['X_train'], data['y_train']")
    print(f"     X_test, y_test = data['X_test'], data['y_test']")
    print(f"     num_features = data['num_features']")
    print("\n   Key improvements:")
    print("     ✓ Chronological 80/20 split (no data leakage)")
    print("     ✓ Delta features (sentiment momentum)")
    print("     ✓ float32 dtype (memory efficient for 5 years of data)")


if __name__ == "__main__":
    output_file = 'twitter_lstm_training_data.npz'
    
    print("=" * 70)
    print("LSTM Training Data Builder - Multi-Ticker")
    print("=" * 70)
    
    print("\n1. Connecting to MongoDB...")
    client = pymongo.MongoClient(MONGO_URI)
    db = client[DB_NAME]
    
    data_dict = build_training_samples(db, prediction_hour=16)
    
    if data_dict is None or len(data_dict['X_train']) == 0:
        print("\n" + "=" * 70)
        print("ERROR: NO TRAINING SAMPLES CREATED")
        print("=" * 70)
        print("\nProblem: Insufficient data density for 72-hour temporal windows.")
        print("\nPossible solutions:")
        print("  1. Increase SAMPLE_SIZE_PER_TICKER in config/db_config.py")
        print("     Current setting: 200 tweets per ticker")
        print("     Recommended: 2000-5000 tweets per ticker for adequate coverage")
        print("\n  2. Disable sampling (set USE_SAMPLE = False) to use full dataset")
        print("\n  3. Ingest additional data from Kaggle source")
        print("\nNote: Sparse tweet data distributed across days/weeks")
        print("cannot provide the required 72 consecutive hours of sentiment features.")
        client.close()
        exit(1)
    
    save_training_data(data_dict, output_file)
    
    client.close()
    
    print("\n" + "=" * 70)
    print("Training data construction complete.")
    print("=" * 70)
