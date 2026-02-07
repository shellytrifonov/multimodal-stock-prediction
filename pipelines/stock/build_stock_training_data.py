import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pymongo
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from config.db_config import MONGO_URI, DB_NAME, LIMIT_TICKERS


def compute_technical_indicators(df):
    """
    Compute technical indicators from RAW price data (NO FUTURE LEAKAGE).
    
    All indicators use only past values. NaN rows from initial windows will be dropped.
    
    Args:
        df (pd.DataFrame): Raw stock data with OHLCV columns
    
    Returns:
        pd.DataFrame: Data with added technical indicators
    """
    df = df.copy()
    
    # Basic returns
    df['return_1d'] = df['close'].pct_change()
    # Handle log of zero/negative returns safely
    with np.errstate(divide='ignore', invalid='ignore'):
        df['log_return_1d'] = np.log(df['close'] / df['close'].shift(1))
    df['log_return_1d'].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Log transform volume to reduce dominance
    df['log_volume'] = np.log1p(df['volume'])
    
    # Momentum (10-day price change)
    df['momentum_10d'] = df['close'] - df['close'].shift(10)
    
    # Simple Moving Averages
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    
    # Exponential Moving Averages
    df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    
    # Volatility (rolling std of returns)
    df['volatility_10'] = df['return_1d'].rolling(window=10).std()
    df['volatility_20'] = df['return_1d'].rolling(window=20).std()
    
    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    # Avoid division by zero: if loss is 0, RSI = 100
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    # Handle edge cases: RSI = 100 when loss = 0, RSI = 0 when gain = 0
    df.loc[loss == 0, 'RSI_14'] = 100
    df.loc[gain == 0, 'RSI_14'] = 0
    df['RSI_14'].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # MACD (Moving Average Convergence Divergence)
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    # Bollinger Bands
    df['BB_mid'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_mid'] + (2 * bb_std)
    df['BB_lower'] = df['BB_mid'] - (2 * bb_std)
    df['BB_width'] = df['BB_upper'] - df['BB_lower']
    
    return df


def load_stock_data_by_ticker(db, ticker):
    """
    Load RAW stock data and compute technical indicators.
    
    Args:
        db: MongoDB database instance
        ticker (str): Stock ticker symbol
    
    Returns:
        pd.DataFrame: Stock data with technical indicators, or None if no data
    """
    cleaned_collection = db['cleaned_stock_prices']
    stock_docs = list(cleaned_collection.find({'ticker': ticker}).sort('date', 1))
    
    if not stock_docs:
        return None
    
    df = pd.DataFrame(stock_docs)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Compute technical indicators on RAW data
    df = compute_technical_indicators(df)
    
    # Replace any remaining inf values with NaN before dropping
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Drop rows with NaN (from initial indicator windows)
    # This ensures all features have valid values
    initial_len = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped = initial_len - len(df)
    if dropped > 0:
        print(f"       Dropped {dropped} initial rows due to indicator warm-up period")
    
    # Final validation: check for any remaining inf/nan
    if df.isnull().any().any() or np.isinf(df.select_dtypes(include=[np.number])).any().any():
        print(f"       WARNING: Still found NaN/inf after dropna()")
        # Fill any remaining with forward fill as last resort
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.replace([np.inf, -np.inf], 0, inplace=True)
    
    return df


def create_sequences(stock_df, sequence_length=60):
    """
    Create sliding window sequences for LSTM training.
    
    Args:
        stock_df (pd.DataFrame): Stock data with features and indicators
        sequence_length (int): Number of days to look back (default: 60)
    
    Returns:
        tuple: (X, y, dates) where:
            - X: numpy array of shape (num_samples, sequence_length, num_features)
            - y: numpy array of binary labels (1=Up, 0=Down)
            - dates: list of prediction dates
    """
    # All features including technical indicators (NO SCALING YET)
    feature_cols = [
        # Original OHLCV features
        'open', 'high', 'low', 'close', 'adj_close',
        # Volume (log-transformed)
        'log_volume',
        # Returns and momentum
        'return_1d', 'log_return_1d', 'momentum_10d',
        # Moving averages
        'SMA_10', 'SMA_20', 'EMA_10', 'EMA_20',
        # Volatility
        'volatility_10', 'volatility_20',
        # RSI
        'RSI_14',
        # MACD
        'MACD', 'MACD_signal', 'MACD_hist',
        # Bollinger Bands
        'BB_mid', 'BB_upper', 'BB_lower', 'BB_width'
    ]
    
    data = stock_df[feature_cols].values
    dates = stock_df['date'].values
    
    X_list = []
    y_list = []
    date_list = []
    
    # Need at least sequence_length + 1 days (sequence + next day for label)
    for i in range(len(data) - sequence_length):
        # Extract sequence of 60 days
        sequence = data[i:i + sequence_length]
        
        # Get current close price (last day of sequence)
        current_close = data[i + sequence_length - 1, 3]  # close is index 3
        
        # Get next day's close price
        next_close = data[i + sequence_length, 3]
        
        # Binary label: 1 if price goes up, 0 if down
        label = 1 if next_close > current_close else 0
        
        X_list.append(sequence)
        y_list.append(label)
        # ✅ FIX: Store CURRENT day (last day of sequence) to match Twitter pipeline
        # This is the day we're OBSERVING, predicting movement to NEXT day
        date_list.append(dates[i + sequence_length - 1])
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    return X, y, date_list


def build_training_samples(db, sequence_length=60):
    """
    Construct LSTM training dataset from stock price sequences.
    
    Important: This function assumes stock data has already been filtered to the
    correct date range by preprocess_stock.py (sentiment data range + 90-day buffer).
    This ensures all generated sequences align with Twitter data periods.
    
    Args:
        db: MongoDB database instance
        sequence_length (int): Lookback window in trading days (default: 60)
    
    Returns:
        tuple: (X, y, dates, tickers) or (None, None, None, None) if no data
    """
    print("\n1. Discovering tickers from database...")
    cleaned_collection = db['cleaned_stock_prices']
    
    if LIMIT_TICKERS:
        tickers = LIMIT_TICKERS
        print(f"   Using specified tickers: {tickers}")
    else:
        tickers = cleaned_collection.distinct('ticker')
        print(f"   Found {len(tickers)} tickers: {tickers}")
    
    if not tickers:
        print("   ERROR: No tickers found in cleaned_stock_prices collection.")
        return None, None, None, None
    
    print(f"\n2. Building training samples...")
    print("=" * 70)
    print(f"   Sequence length: {sequence_length} trading days")
    print(f"   Features per day: 25 (OHLCV + 20 technical indicators)")
    print(f"   Target: Binary classification (Next day Up=1, Down=0)")
    print(f"\n   Technical indicators included:")
    print(f"     • Returns: return_1d, log_return_1d, momentum_10d")
    print(f"     • Moving Averages: SMA_10, SMA_20, EMA_10, EMA_20")
    print(f"     • Volatility: volatility_10, volatility_20")
    print(f"     • RSI_14, MACD, MACD_signal, MACD_hist")
    print(f"     • Bollinger Bands: BB_mid, BB_upper, BB_lower, BB_width")
    
    X_list = []
    y_list = []
    date_list = []
    ticker_list = []
    total_skipped = 0
    
    for ticker in tickers:
        print(f"\n   Processing {ticker}...")
        
        stock_df = load_stock_data_by_ticker(db, ticker)
        
        if stock_df is None:
            print(f"     No stock data for {ticker}")
            continue
        
        print(f"     Trading days available: {len(stock_df)}")
        
        # Check if we have enough data
        if len(stock_df) < sequence_length + 1:
            print(f"     WARNING: Insufficient data (need {sequence_length + 1} days, have {len(stock_df)})")
            total_skipped += 1
            continue
        
        # Create sequences
        X, y, dates = create_sequences(stock_df, sequence_length)
        
        if len(X) == 0:
            print(f"     No sequences created for {ticker}")
            total_skipped += 1
            continue
        
        # Add to lists
        X_list.extend(X)
        y_list.extend(y)
        date_list.extend(dates)
        ticker_list.extend([ticker] * len(X))
        
        print(f"     Sequences created: {len(X)}")
        print(f"     Date range: {stock_df['date'].min().date()} to {stock_df['date'].max().date()}")
        print(f"     Labels - Up: {np.sum(y)} ({100*np.mean(y):.1f}%), Down: {len(y) - np.sum(y)} ({100*(1-np.mean(y)):.1f}%)")
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    dates = np.array(date_list)
    tickers = np.array(ticker_list)
    
    print(f"\n3. Implementing chronological 80/20 train/test split...")
    print("=" * 70)
    
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
    
    # CRITICAL: Apply scaling AFTER split to prevent data leakage
    print(f"\n4. Applying StandardScaler (fit on train only)...")
    print("=" * 70)
    
    # Get original shape
    n_train, seq_len, n_features = X_train.shape
    n_test = X_test.shape[0]
    
    # Reshape to 2D for scaling: (n_samples * seq_len, n_features)
    X_train_2d = X_train.reshape(-1, n_features)
    X_test_2d = X_test.reshape(-1, n_features)
    
    # Final validation before scaling
    if np.isinf(X_train_2d).any() or np.isnan(X_train_2d).any():
        print(f"   WARNING: Found inf/nan in train data before scaling")
        print(f"   Inf count: {np.isinf(X_train_2d).sum()}")
        print(f"   NaN count: {np.isnan(X_train_2d).sum()}")
        # Replace with 0 as last resort
        X_train_2d = np.nan_to_num(X_train_2d, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_2d = np.nan_to_num(X_test_2d, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Fit scaler ONLY on training data
    scaler = StandardScaler()
    X_train_scaled_2d = scaler.fit_transform(X_train_2d)
    
    # Transform test data using training statistics (NO LEAKAGE)
    X_test_scaled_2d = scaler.transform(X_test_2d)
    
    # Reshape back to 3D
    X_train = X_train_scaled_2d.reshape(n_train, seq_len, n_features)
    X_test = X_test_scaled_2d.reshape(n_test, seq_len, n_features)
    
    print(f"   Scaler fitted on {n_train} train samples")
    print(f"   Train mean: {scaler.mean_[:5]}... (first 5 features)")
    print(f"   Train std: {scaler.scale_[:5]}... (first 5 features)")
    print(f"   ✓ Scaling complete (NO DATA LEAKAGE)")
    
    # Save scaler for future use
    scaler_path = 'stock_scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"   ✓ Scaler saved to {scaler_path}")
    
    print(f"\n5. Results:")
    print("=" * 70)
    print(f"   Tickers processed: {len(tickers) - total_skipped}")
    print(f"   Tickers skipped: {total_skipped}")
    print(f"\n   Train set:")
    print(f"     Shape: {X_train.shape} (samples, sequence_length, features)")
    print(f"     Up: {np.sum(y_train)} ({100*np.mean(y_train):.1f}%)")
    print(f"     Down: {len(y_train) - np.sum(y_train)} ({100*(1-np.mean(y_train)):.1f}%)")
    print(f"\n   Test set:")
    print(f"     Shape: {X_test.shape} (samples, sequence_length, features)")
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
        'tickers_test': tickers_test
    }


def save_training_data(data_dict, output_file):
    """
    Save training dataset with pre-split train/test to compressed NumPy archive.
    
    Args:
        data_dict (dict): Dictionary containing train/test splits
        output_file (str): Output filename
    """
    np.savez_compressed(output_file,
                       X_train=data_dict['X_train'],
                       y_train=data_dict['y_train'],
                       dates_train=[pd.Timestamp(d).strftime('%Y-%m-%d') for d in data_dict['dates_train']],
                       tickers_train=data_dict['tickers_train'],
                       X_test=data_dict['X_test'],
                       y_test=data_dict['y_test'],
                       dates_test=[pd.Timestamp(d).strftime('%Y-%m-%d') for d in data_dict['dates_test']],
                       tickers_test=data_dict['tickers_test'])
    
    file_size_mb = os.path.getsize(output_file) / 1024 / 1024
    print(f"\n6. Saved: {output_file}")
    print("=" * 70)
    print(f"   File size: {file_size_mb:.2f} MB")
    print("\n   Loading instructions:")
    print(f"     data = np.load('{output_file}')")
    print(f"     X_train, y_train = data['X_train'], data['y_train']")
    print(f"     X_test, y_test = data['X_test'], data['y_test']")
    print("\n   Key improvements:")
    print("     ✓ 25 features (OHLCV + 20 technical indicators)")
    print("     ✓ StandardScaler fitted on train only (NO DATA LEAKAGE)")
    print("     ✓ Chronological 80/20 split")
    print("     ✓ Scaler saved to stock_scaler.pkl for reproducibility")


if __name__ == "__main__":
    output_file = 'stock_lstm_training_data.npz'
    
    print("=" * 70)
    print("Stock LSTM Training Data Builder")
    print("=" * 70)
    
    print("\n1. Connecting to MongoDB...")
    client = pymongo.MongoClient(MONGO_URI)
    db = client[DB_NAME]
    
    # Build training samples with 60-day sequences
    data_dict = build_training_samples(db, sequence_length=60)
    
    if data_dict is None or len(data_dict['X_train']) == 0:
        print("\n" + "=" * 70)
        print("ERROR: NO TRAINING SAMPLES CREATED")
        print("=" * 70)
        print("\nProblem: Insufficient stock price data.")
        print("\nPossible solutions:")
        print("  1. Ensure you have run preprocess_stock.py first")
        print("  2. Check that cleaned_stock_prices collection has data")
        print("  3. Verify tickers have at least 61 consecutive trading days")
        print("  4. Run data ingestion to fetch stock prices from Yahoo Finance")
        print(f"\nRequired: At least 61 trading days per ticker (60 for sequence + 1 for label)")
        client.close()
        exit(1)
    
    save_training_data(data_dict, output_file)
    
    client.close()
    
    print("\n" + "=" * 70)
    print("Training data construction complete.")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Train the Stock LSTM model: python models/train_stock_lstm.py")
    print("  2. Verify date alignment with sentiment data before fusion training")
    print("\nDate Alignment Strategy:")
    print("  • Stock data range = Sentiment range + 90-day buffer (BEFORE min date)")
    print("  • 60-day sequences ensure LSTM has historical context")
    print("  • Prediction dates align with Twitter availability for fusion")
