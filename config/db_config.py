# MongoDB connection parameters
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "StockPredictionDB"

START_DATE = "2015-01-01"
END_DATE = "2020-01-01"


# Twitter Pipeline Collections
COLLECTION_RAW_TWEETS = "raw_tweets"  # Stage 1: Raw data from Kaggle dataset
COLLECTION_CLEANED_TWEETS = "cleaned_tweets"  # Stage 2: Text preprocessed and cleaned
COLLECTION_TWEETS_WITH_SENTIMENT = "tweets_with_sentiment"  # Stage 3: Sentiment scores computed
COLLECTION_HOURLY_SENTIMENT_TWEETS = "hourly_sentiment_tweets"  # Stage 4: Temporal aggregation (hourly)

# Stock Price Data
COLLECTION_STOCK_PRICES = "stock_prices"   # Historical stock price data from Yahoo Finance

USE_SAMPLE = False
SAMPLE_SIZE_PER_TICKER = None
LIMIT_TICKERS = ['AAPL', 'AMZN', 'GOOG', 'GOOGL', 'MSFT', 'TSLA']