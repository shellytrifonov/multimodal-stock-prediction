import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pymongo
from config.db_config import (
    MONGO_URI, DB_NAME,
    COLLECTION_RAW_TWEETS, COLLECTION_CLEANED_TWEETS,
    COLLECTION_TWEETS_WITH_SENTIMENT, COLLECTION_HOURLY_SENTIMENT_TWEETS,
    COLLECTION_NEWS_ARTICLES, COLLECTION_CLEANED_NEWS,
    COLLECTION_NEWS_WITH_SENTIMENT, COLLECTION_HOURLY_NEWS_SENTIMENT,
    COLLECTION_STOCK_PRICES
)

def create_indexes():
    """Create MongoDB indexes to optimize query performance."""
    print("=" * 70)
    print("MongoDB Index Creation")
    print("=" * 70)
    
    print("\nConnecting to MongoDB...")
    client = pymongo.MongoClient(MONGO_URI)
    db = client[DB_NAME]
    
    indexes_created = 0
    
    standard_collections = [
        COLLECTION_RAW_TWEETS,
        COLLECTION_CLEANED_TWEETS,
        COLLECTION_TWEETS_WITH_SENTIMENT,
        COLLECTION_NEWS_ARTICLES,
        COLLECTION_CLEANED_NEWS,
        COLLECTION_NEWS_WITH_SENTIMENT
    ]
    
    print("\n1. Creating single-field indexes on 'ticker'...")
    for collection_name in standard_collections:
        collection = db[collection_name]
        try:
            result = collection.create_index([("ticker", pymongo.ASCENDING)])
            print(f"   ✓ {collection_name}: ticker_1")
            indexes_created += 1
        except Exception as e:
            print(f"   ✗ {collection_name}: {e}")
    
    print("\n2. Creating compound indexes for temporal queries...")
    try:
        collection = db[COLLECTION_HOURLY_SENTIMENT_TWEETS]
        result = collection.create_index([
            ("ticker", pymongo.ASCENDING),
            ("Hour", pymongo.ASCENDING)
        ])
        print(f"   ✓ {COLLECTION_HOURLY_SENTIMENT_TWEETS}: ticker_1_Hour_1")
        indexes_created += 1
    except Exception as e:
        print(f"   ✗ {COLLECTION_HOURLY_SENTIMENT_TWEETS}: {e}")
    
    try:
        collection = db[COLLECTION_HOURLY_NEWS_SENTIMENT]
        result = collection.create_index([
            ("ticker", pymongo.ASCENDING),
            ("Hour", pymongo.ASCENDING)
        ])
        print(f"   ✓ {COLLECTION_HOURLY_NEWS_SENTIMENT}: ticker_1_Hour_1")
        indexes_created += 1
    except Exception as e:
        print(f"   ✗ {COLLECTION_HOURLY_NEWS_SENTIMENT}: {e}")
    
    try:
        collection = db[COLLECTION_STOCK_PRICES]
        result = collection.create_index([
            ("ticker", pymongo.ASCENDING),
            ("date", pymongo.ASCENDING)
        ])
        print(f"   ✓ {COLLECTION_STOCK_PRICES}: ticker_1_date_1")
        indexes_created += 1
    except Exception as e:
        print(f"   ✗ {COLLECTION_STOCK_PRICES}: {e}")
    
    print("\n3. Creating timestamp indexes for date filtering...")
    try:
        collection = db[COLLECTION_TWEETS_WITH_SENTIMENT]
        result = collection.create_index([("created_at", pymongo.ASCENDING)])
        print(f"   ✓ {COLLECTION_TWEETS_WITH_SENTIMENT}: created_at_1")
        indexes_created += 1
    except Exception as e:
        print(f"   ✗ {COLLECTION_TWEETS_WITH_SENTIMENT}: {e}")
    
    try:
        collection = db[COLLECTION_NEWS_WITH_SENTIMENT]
        result = collection.create_index([("published_at", pymongo.ASCENDING)])
        print(f"   ✓ {COLLECTION_NEWS_WITH_SENTIMENT}: published_at_1")
        indexes_created += 1
    except Exception as e:
        print(f"   ✗ {COLLECTION_NEWS_WITH_SENTIMENT}: {e}")
    
    print("\n" + "=" * 70)
    print(f"Index creation complete: {indexes_created} indexes created")
    print("=" * 70)
    
    print("\n4. Verifying indexes...")
    for collection_name in [COLLECTION_HOURLY_SENTIMENT_TWEETS, 
                           COLLECTION_HOURLY_NEWS_SENTIMENT, 
                           COLLECTION_STOCK_PRICES]:
        collection = db[collection_name]
        indexes = list(collection.list_indexes())
        print(f"\n   {collection_name}:")
        for idx in indexes:
            print(f"     - {idx['name']}")
    
    client.close()
    print("\n✓ Database optimization complete")


if __name__ == "__main__":
    create_indexes()
