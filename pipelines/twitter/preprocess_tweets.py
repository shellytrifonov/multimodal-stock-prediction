import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pymongo
import re
import html
from config.db_config import (
    MONGO_URI, DB_NAME, 
    COLLECTION_RAW_TWEETS, COLLECTION_CLEANED_TWEETS
)

INSERT_BATCH_SIZE = 5000

print("=" * 70)
print("Preprocessing Tweets")
print("=" * 70)

def detect_language(text):
    """Heuristic English detection optimized for financial text."""
    if not text or len(text) < 3:
        return False
    
    english_words = {
        # --- Common Stop Words ---
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
        'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
        'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
        'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
        'is', 'are', 'was', 'were', 'been', 'has', 'had', 'does', 'did', 'can',
        'could', 'should', 'about', 'after', 'over', 'under', 'up', 'down',
        
        # --- Core Market Terms ---
        'stock', 'market', 'price', 'buy', 'sell', 'trading', 'trade',
        'shares', 'earnings', 'revenue', 'profit', 'loss', 'quarter', 'reports',
        'hold', 'long', 'short', 'call', 'put', 'bull', 'bear', 'rally', 
        'surge', 'plunge', 'dip', 'gap', 'eps', 'guidance', 'forecast',
        
        # --- Macro Economics ---
        'inflation', 'rate', 'rates', 'fed', 'federal', 'bank', 'economy', 
        'recession', 'gdp', 'cpi', 'debt', 'bond', 'yield', 'crypto', 'bitcoin',
        
        # --- Corporate Actions ---
        'dividend', 'split', 'merger', 'acquisition', 'deal', 'ipo', 'ceo', 'cfo',
        'board', 'investor', 'shareholder', 'analyst', 'target', 'rating', 'upgrade', 'downgrade',
        
        # --- Movement Verbs/Adjectives ---
        'record', 'high', 'low', 'gain', 'drop', 'crash', 'boom', 'correction',
        'volatility', 'volume', 'green', 'red', 'support', 'resistance', 'breakout'
    }
    
    words = re.findall(r'\b[a-z]+\b', text.lower())
    
    if len(words) < 5:
        return any(word in english_words for word in words)
    
    english_count = sum(1 for word in words if word in english_words)
    return (english_count / len(words)) >= 0.3


def remove_urls(text):
    """Remove URLs from text."""
    text = re.sub(r'http\S+|www\.\S+', '', text)
    return text


def remove_mentions_and_hashtags(text):
    """Remove @mentions and #hashtags."""
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    return text


def remove_emojis(text):
    """Remove emojis from text."""
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)


def clean_whitespace(text):
    """Remove extra whitespace."""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def preprocess_text(text):
    """Apply text cleaning pipeline to financial tweets."""

    # Ensure text is not empty and is a string
    if not text or not isinstance(text, str):
        return None
    
    text = html.unescape(text) # Decode HTML entities
    text = re.sub(r'^RT[\s]+', '', text) # Remove 'RT' (Retweet) marker from start of text
    text = remove_urls(text) # Remove URLs
    text = re.sub(r'\b0x[a-fA-F0-9]{40}\b', '', text) # Remove Ethereum wallet addresses (start with 0x)
    text = re.sub(r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b', '', text) # Remove Bitcoin wallet addresses (start with 1 or 3)
    text = remove_mentions_and_hashtags(text) # Remove @mentions and #hashtags
    text = remove_emojis(text) # Remove emojis
    text = re.sub(r'([!?.,()])', r' \1 ', text) # Add spacing around punctuation
    text = clean_whitespace(text) # Clean up multiple spaces and trimming
    
    # Discard texts shorter than 10 characters
    if len(text) < 10:
        return None
    
    # Discard texts with fewer than 3 words
    words = text.split()
    if len(words) < 3:
        return None
        
    return text


def preprocess_tweets():
    """Execute preprocessing pipeline on tweet corpus."""
    print("\n1. Connecting to MongoDB...")
    client = pymongo.MongoClient(MONGO_URI)
    db = client[DB_NAME]
    raw_collection = db[COLLECTION_RAW_TWEETS]
    cleaned_collection = db[COLLECTION_CLEANED_TWEETS]
    
    print("   Clearing previous cleaned tweets...")
    cleaned_collection.delete_many({})
    print("   Ready for preprocessing.")
    
    print("\n2. Discovering tickers from database...")
    tickers = raw_collection.distinct('ticker')
    print(f"   Found {len(tickers)} tickers: {tickers}")
    
    if not tickers:
        print("   ERROR: No tickers found. Run data ingestion first.")
        client.close()
        return
    
    print(f"\n3. Preprocessing tweets...")
    print("=" * 70)
    
    total_processed = 0
    total_filtered = 0
    total_cleaned = 0
    
    for ticker in tickers:
        print(f"\n   Processing {ticker}...")
        
        query = {'ticker': ticker}
        tweets = list(raw_collection.find(query))
        
        if not tweets:
            print(f"     No tweets found for {ticker}")
            continue
        
        print(f"     Found {len(tweets)} raw tweets")
        
        filtered = 0
        cleaned = 0
        cleaned_docs = []
        batch_count = 0
        
        for tweet in tweets:
            total_processed += 1
            original_text = tweet.get('text', '')
            
            if not detect_language(original_text):
                filtered += 1
                continue
            
            cleaned_text = preprocess_text(original_text)
            
            if cleaned_text is None:
                filtered += 1
            else:
                cleaned_doc = {
                    'ticker': tweet['ticker'],
                    'platform': tweet.get('platform', 'Twitter'),
                    'post_id': tweet['post_id'],
                    'original_text': original_text,
                    'cleaned_text': cleaned_text,
                    'author': tweet.get('author'),
                    'created_at': tweet['created_at'],
                    'likes_count': tweet.get('likes_count', 0),
                    'retweet_count': tweet.get('retweet_count', 0),
                    'comment_count': tweet.get('comment_count', 0)
                }
                cleaned_docs.append(cleaned_doc)
                cleaned += 1
                
                if len(cleaned_docs) >= INSERT_BATCH_SIZE:
                    cleaned_collection.insert_many(cleaned_docs)
                    batch_count += 1
                    print(f"     Batch {batch_count}: Inserted {len(cleaned_docs)} documents")
                    cleaned_docs = []
        
        if cleaned_docs:
            cleaned_collection.insert_many(cleaned_docs)
            batch_count += 1
            print(f"     Final batch: Inserted {len(cleaned_docs)} documents")
        
        total_filtered += filtered
        total_cleaned += cleaned
        
        print(f"     Cleaned: {cleaned}, Filtered: {filtered}")
    
    print(f"\n4. Summary:")
    print("=" * 70)
    print(f"   Total tweets processed: {total_processed}")
    print(f"   Cleaned (English): {total_cleaned}")
    print(f"   Filtered out: {total_filtered}")
    print(f"   Success rate: {100 * total_cleaned / total_processed:.1f}%")
    print(f"\n   Preprocessing complete.")
    
    client.close()


if __name__ == "__main__":
    preprocess_tweets()
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
