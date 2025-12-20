import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pymongo
import pandas as pd
import numpy as np
from datetime import datetime
from config.db_config import (
    MONGO_URI, DB_NAME,
    COLLECTION_NEWS_WITH_SENTIMENT, COLLECTION_HOURLY_NEWS_SENTIMENT
)


def aggregate_by_hour():
    """Perform hourly aggregation of news sentiment data with statistical features."""
    print("=" * 70)
    print("Hourly News Sentiment Aggregation")
    print("=" * 70)
    
    print("\n1. Connecting to MongoDB...")
    client = pymongo.MongoClient(MONGO_URI)
    db = client[DB_NAME]
    sentiment_collection = db[COLLECTION_NEWS_WITH_SENTIMENT]
    hourly_collection = db[COLLECTION_HOURLY_NEWS_SENTIMENT]
    
    print("\n2. Discovering tickers from database...")
    tickers = sentiment_collection.distinct('ticker')
    print(f"   Found {len(tickers)} tickers: {tickers}")
    
    if len(tickers) == 0:
        print("   ERROR: No tickers found in news_with_sentiment collection.")
        print("   Please run news_sentiment.py first.")
        client.close()
        return
    
    total_articles = sentiment_collection.count_documents({})
    print(f"   Total articles with sentiment: {total_articles:,}")
    
    print("\n3. Clearing old hourly data...")
    hourly_collection.delete_many({})
    
    print("\n4. Aggregating by hour for each ticker...")
    total_documents = 0
    
    for ticker in tickers:
        print(f"\n   Processing {ticker}...")
        
        query = {'ticker': ticker}
        articles = list(sentiment_collection.find(query))
        
        if len(articles) == 0:
            print(f"     No articles with sentiment for {ticker}")
            continue
        
        print(f"     Found {len(articles)} articles")
        
        df = pd.DataFrame(articles)
        df['published_at'] = pd.to_datetime(df['published_at'])
        
        weekend_mask = df['published_at'].dt.dayofweek >= 5
        
        if weekend_mask.any():
            days_to_monday = 7 - df.loc[weekend_mask, 'published_at'].dt.dayofweek
            next_monday = df.loc[weekend_mask, 'published_at'].dt.normalize() + pd.to_timedelta(days_to_monday, unit='D')
            next_monday = next_monday + pd.Timedelta(hours=8)
            df.loc[weekend_mask, 'published_at'] = next_monday
            
            weekend_count = weekend_mask.sum()
            print(f"     Rolled {weekend_count} weekend articles to Monday 8AM")
        
        df['Hour'] = df['published_at'].dt.floor('h')
        
        hourly = df.groupby('Hour').agg({
            'p_pos': 'mean',
            'p_neg': 'mean',
            'p_neu': 'mean',
            'sentiment_score': ['mean', 'count', 'std', 'max', 'min']
        })
        
        hourly.columns = ['_'.join(str(c) for c in col).strip('_') if isinstance(col, tuple) else col 
                          for col in hourly.columns.values]
        col_mapping = {
            'p_pos_mean': 'p_pos',
            'p_neg_mean': 'p_neg',
            'p_neu_mean': 'p_neu',
            'sentiment_score_mean': 'mean_sentiment',
            'sentiment_score_count': 'news_count',
            'sentiment_score_std': 'sentiment_std',
            'sentiment_score_max': 'max_sentiment',
            'sentiment_score_min': 'min_sentiment'
        }
        hourly = hourly.rename(columns=col_mapping)
        hourly = hourly.reset_index()
        
        first_hour = hourly['Hour'].min()
        last_hour = hourly['Hour'].max()
        all_hours = pd.date_range(start=first_hour, end=last_hour, freq='h')
        
        complete_df = pd.DataFrame({'Hour': all_hours})
        result = complete_df.merge(hourly, on='Hour', how='left')
        result['p_pos'] = result['p_pos'].fillna(0)
        result['p_neg'] = result['p_neg'].fillna(0)
        result['p_neu'] = result['p_neu'].fillna(0)
        
        result['mean_sentiment'] = result['mean_sentiment'].fillna(0)
        result['news_count'] = result['news_count'].fillna(0).astype(int)
        result['sentiment_std'] = result['sentiment_std'].fillna(0)
        result['max_sentiment'] = result['max_sentiment'].fillna(0)
        result['min_sentiment'] = result['min_sentiment'].fillna(0)
        
        documents = []
        for _, row in result.iterrows():
            doc = {
                'hour': row['Hour'],
                'ticker': ticker,
                'p_pos': float(row['p_pos']),
                'p_neg': float(row['p_neg']),
                'p_neu': float(row['p_neu']),
                'mean_sentiment': float(row['mean_sentiment']),
                'news_count': int(row['news_count']),
                'sentiment_std': float(row['sentiment_std']),
                'max_sentiment': float(row['max_sentiment']),
                'min_sentiment': float(row['min_sentiment'])
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
