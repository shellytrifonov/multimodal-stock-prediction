import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import pandas as pd
import pymongo
from datetime import datetime, timedelta
from config.db_config import MONGO_URI, DB_NAME, COLLECTION_STOCK_PRICES, COLLECTION_RAW_TWEETS, LIMIT_TICKERS

print("=" * 70)
print("Fetching Stock Prices from Yahoo Finance")
print("=" * 70)

# Connect to MongoDB
client = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]
raw_tweets_collection = db[COLLECTION_RAW_TWEETS]
stock_collection = db[COLLECTION_STOCK_PRICES]

# Discover tickers
if LIMIT_TICKERS:
    tickers = LIMIT_TICKERS
    print(f"Using specified tickers: {tickers}")
else:
    tickers = raw_tweets_collection.distinct('ticker')
    print(f"Found {len(tickers)} tickers in raw tweets: {tickers}")

if not tickers:
    print("ERROR: No tickers found. Run ingest_kaggle_data.py first.")
    client.close()
    exit(1)

# Clear old stock data
if LIMIT_TICKERS:
    stock_collection.delete_many({'ticker': {'$in': LIMIT_TICKERS}})
else:
    stock_collection.delete_many({})

# Fetch stock data for each ticker
total_saved = 0

for ticker in tickers:
    print(f"\nProcessing {ticker}...")
    
    # Get date range from tweets
    ticker_docs = list(raw_tweets_collection.find({'ticker': ticker}).sort('created_at', 1))
    if not ticker_docs:
        print(f"No tweets for {ticker}")
        continue
    
    first_date = pd.to_datetime(ticker_docs[0]['created_at'])
    last_date = pd.to_datetime(ticker_docs[-1]['created_at'])
    
    start_date = (first_date - timedelta(days=100)).strftime('%Y-%m-%d')
    end_date = (last_date + timedelta(days=3)).strftime('%Y-%m-%d')
    
    print(f"Date range: {start_date} to {end_date}")
    
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            print(f"WARNING: No data available from Yahoo Finance, skipping {ticker}")
            continue
        
        df = df.reset_index()
        
        # Normalize column names
        df.columns = [c.capitalize() for c in df.columns]
        
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [c for c in required_cols if c not in df.columns]
        
        if missing_cols:
            print(f"   WARNING: Missing columns {missing_cols}. Available: {df.columns}")
            continue

        # Standardize date format
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        
        print(f"     Downloaded {len(df)} trading days")
        
        documents = []
        for _, row in df.iterrows():
            doc = {
                'ticker': ticker,
                'date': row['Date'],
                'open': float(row['Open']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'close': float(row['Close']),
                'volume': float(row['Volume']),
                # Try to get Adj Close if available, else use Close
                'adj_close': float(row.get('Adj Close', row['Close']))
            }
            documents.append(doc)
        
        if documents:
            stock_collection.insert_many(documents)
            total_saved += len(documents)
            print(f"     Saved {len(documents)} prices for {ticker}")
    
    except Exception as e:
        print(f"     ERROR: Failed to fetch {ticker}: {e}")
        continue

# Create compound index
print(f"\n5. Creating indexes...")
stock_collection.create_index([('ticker', 1), ('date', 1)])

print("\n6. Summary:")
print("=" * 70)
print(f"   Tickers processed: {len(tickers)}")
print(f"   Total stock prices saved: {total_saved}")
print(f"   Stock data fetch complete.")

client.close()

print("\n" + "=" * 70)
print("Done!")
print("=" * 70)