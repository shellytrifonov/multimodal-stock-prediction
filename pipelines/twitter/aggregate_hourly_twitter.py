import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pymongo
import pandas as pd
import numpy as np
from datetime import datetime
from config.db_config import (
    MONGO_URI, DB_NAME,
    COLLECTION_TWEETS_WITH_SENTIMENT, COLLECTION_HOURLY_SENTIMENT_TWEETS
)

def aggregate_by_hour():
    """Perform hourly aggregation of tweet sentiment data with statistical features."""
    print("=" * 70)
    print("Hourly Sentiment Aggregation")
    print("=" * 70)
    
    print("\n1. Connecting to MongoDB...")
    client = pymongo.MongoClient(MONGO_URI)
    db = client[DB_NAME]
    sentiment_collection = db[COLLECTION_TWEETS_WITH_SENTIMENT]
    hourly_collection = db[COLLECTION_HOURLY_SENTIMENT_TWEETS]
    
    print("\n2. Discovering tickers from database...")
    tickers = sentiment_collection.distinct('ticker')
    print(f"   Found {len(tickers)} tickers: {tickers}")
    
    if len(tickers) == 0:
        print("   ERROR: No tickers found in tweets_with_sentiment collection.")
        print("   Please run twitter_sentiment.py first.")
        client.close()
        return
    
    total_tweets = sentiment_collection.count_documents({})
    print(f"   Total tweets with sentiment: {total_tweets:,}")
    
    print("\n3. Clearing old hourly data...")
    hourly_collection.delete_many({})
    
    print("\n4. Aggregating by hour for each ticker...")
    total_documents = 0
    
    for ticker in tickers:
        print(f"\n   Processing {ticker}...")
        
        query = {'ticker': ticker}
        tweets = list(sentiment_collection.find(query))
        
        if len(tweets) == 0:
            print(f"     No tweets with sentiment for {ticker}")
            continue
        
        print(f"     Found {len(tweets)} tweets")
        
        df = pd.DataFrame(tweets)
        df['created_at'] = pd.to_datetime(df['created_at'])
        
        # Calculate engagement weight using log scale to prevent outliers
        df['engagement'] = df['likes_count'].fillna(0) + df['retweet_count'].fillna(0)
        df['weight'] = np.log1p(df['engagement'])  # log(1 + engagement)
                
        weekend_mask = df['created_at'].dt.dayofweek >= 5
        
        if weekend_mask.any():
            days_to_monday = 7 - df.loc[weekend_mask, 'created_at'].dt.dayofweek
            next_monday = df.loc[weekend_mask, 'created_at'].dt.normalize() + pd.to_timedelta(days_to_monday, unit='D')
            next_monday = next_monday + pd.Timedelta(hours=8)
            df.loc[weekend_mask, 'created_at'] = next_monday
            
            weekend_count = weekend_mask.sum()
            print(f"     Rolled {weekend_count} weekend tweets to Monday 8AM")
        
        df['Hour'] = df['created_at'].dt.floor('h')
        
        # Calculate weighted averages for each hour
        def weighted_avg(group):
            weights = group['weight']
            # Normalize weights within this hour
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                # If all weights are 0, use equal weighting
                weights = np.ones(len(group)) / len(group)
            
            return pd.Series({
                'p_negative': (group['p_negative'] * weights).sum(),
                'p_neutral': (group['p_neutral'] * weights).sum(),
                'p_positive': (group['p_positive'] * weights).sum(),
                'mean_sentiment': (group['sentiment_score'] * weights).sum(),
                'tweet_count': len(group),
                'sentiment_std': group['sentiment_score'].std(),
                'max_sentiment': group['sentiment_score'].max(),
                'min_sentiment': group['sentiment_score'].min(),
                'total_engagement': group['engagement'].sum(),
                'avg_engagement': group['engagement'].mean()
            })
        
        hourly = df.groupby('Hour').apply(weighted_avg).reset_index()
        
        # Column names are already set by the weighted_avg function
        
        first_hour = hourly['Hour'].min()
        last_hour = hourly['Hour'].max()
        all_hours = pd.date_range(start=first_hour, end=last_hour, freq='h')
        
        complete_df = pd.DataFrame({'Hour': all_hours})
        result = complete_df.merge(hourly, on='Hour', how='left')
        result['p_negative'] = result['p_negative'].fillna(0)
        result['p_neutral'] = result['p_neutral'].fillna(0)
        result['p_positive'] = result['p_positive'].fillna(0)
        result['mean_sentiment'] = result['mean_sentiment'].fillna(0)
        result['tweet_count'] = result['tweet_count'].fillna(0).astype(int)
        result['sentiment_std'] = result['sentiment_std'].fillna(0)
        result['max_sentiment'] = result['max_sentiment'].fillna(0)
        result['min_sentiment'] = result['min_sentiment'].fillna(0)
        result['total_engagement'] = result['total_engagement'].fillna(0)
        result['avg_engagement'] = result['avg_engagement'].fillna(0)
        
        documents = []
        for _, row in result.iterrows():
            doc = {
                'hour': row['Hour'],
                'ticker': ticker,
                'p_negative': float(row['p_negative']),
                'p_neutral': float(row['p_neutral']),
                'p_positive': float(row['p_positive']),
                'mean_sentiment': float(row['mean_sentiment']),
                'tweet_count': int(row['tweet_count']),
                'sentiment_std': float(row['sentiment_std']),
                'max_sentiment': float(row['max_sentiment']),
                'min_sentiment': float(row['min_sentiment']),
                'total_engagement': float(row['total_engagement']),
                'avg_engagement': float(row['avg_engagement'])
            }
            documents.append(doc)
        
        if documents:
            hourly_collection.insert_many(documents)
            total_documents += len(documents)
        
        print(f"     Saved {len(documents)} hourly records for {ticker}")
    
    print("\n5. Summary:")
    print("=" * 70)
    print(f"   Tickers processed: {len(tickers)}")
    print(f"   Total hourly records: {total_documents}")
    print(f"   Aggregation complete.")
    
    client.close()


if __name__ == "__main__":
    aggregate_by_hour()
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
