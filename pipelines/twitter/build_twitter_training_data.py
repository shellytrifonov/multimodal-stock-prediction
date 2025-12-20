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
    """Extract 72-hour sentiment feature window."""
    end_time = prediction_time - timedelta(hours=1)
    start_time = prediction_time - timedelta(hours=72)
    hour_range = pd.date_range(start=start_time, end=end_time, freq='h')
    
    if len(hour_range) != 72:
        return None
    
    feature_cols = sentiment_df.columns.tolist()
    num_features = len(feature_cols)
    
    window_data = []
    for hour in hour_range:
        if hour in sentiment_df.index:
            row_values = sentiment_df.loc[hour, feature_cols].values
            window_data.append(row_values)
        else:
            window_data.append(np.zeros(num_features))
    
    return np.array(window_data)


def build_training_samples(db, prediction_hour=16):
    """Construct LSTM training dataset by aligning sentiment windows with price labels."""
    print("\n1. Discovering tickers from database...")
    hourly_collection = db[COLLECTION_HOURLY_SENTIMENT_TWEETS]
    
    tickers = hourly_collection.distinct('ticker')
    print(f"   Found {len(tickers)} tickers: {tickers}")
    
    if not tickers:
        print("   ERROR: No tickers found in hourly_sentiment_tweets collection.")
        return None, None, None, None
    
    print(f"\n2. Building training samples...")
    print("=" * 70)
    print(f"   Prediction time: {prediction_hour}:00 each trading day")
    print(f"   Sentiment window: 72 hours x 8 features")
    print(f"   Features: [p_neg, p_neu, p_pos, mean_sent, count, std, max, min]")
    
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
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"\n3. Results:")
    print("=" * 70)
    print(f"   Tickers processed: {len(tickers)}")
    print(f"   Total samples created: {len(X)}")
    print(f"   Total samples skipped: {total_skipped}")
    print(f"\n   Output shapes:")
    print(f"     X: {X.shape} (samples, timesteps, features)")
    print(f"     y: {y.shape} (samples,)")
    print(f"\n   Label distribution:")
    print(f"     Up (1): {np.sum(y)} ({100*np.mean(y):.1f}%)")
    print(f"     Down (0): {len(y) - np.sum(y)} ({100*(1-np.mean(y)):.1f}%)")
    
    return X, y, date_list, ticker_list


def save_training_data(X, y, dates, tickers, output_file):
    """Save training dataset to compressed NumPy archive."""
    np.savez(output_file,
             X=X,
             y=y,
             dates=[d.strftime('%Y-%m-%d') for d in dates],
             tickers=tickers)
    
    num_features = X.shape[2]
    print(f"\n4. Saved: {output_file}")
    print("=" * 70)
    print("\n   Loading instructions:")
    print(f"     data = np.load('{output_file}')")
    print(f"     X = data['X']  # Shape: (samples, 72, {num_features})")
    print("     y = data['y']  # Shape: (samples,)")
    print("     tickers = data['tickers']  # Ticker symbol for each sample")


if __name__ == "__main__":
    output_file = 'twitter_lstm_training_data.npz'
    
    print("=" * 70)
    print("LSTM Training Data Builder - Multi-Ticker")
    print("=" * 70)
    
    print("\n1. Connecting to MongoDB...")
    client = pymongo.MongoClient(MONGO_URI)
    db = client[DB_NAME]
    
    X, y, dates, tickers = build_training_samples(db, prediction_hour=16)
    
    if X is None or len(X) == 0:
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
    
    save_training_data(X, y, dates, tickers, output_file)
    
    client.close()
    
    print("\n" + "=" * 70)
    print("Training data construction complete.")
    print("=" * 70)
