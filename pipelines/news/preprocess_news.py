import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pymongo
import re
import html
from config.db_config import (
    MONGO_URI, DB_NAME, 
    COLLECTION_NEWS_ARTICLES, COLLECTION_CLEANED_NEWS
)

INSERT_BATCH_SIZE = 5000

print("=" * 70)
print("Preprocessing News Headlines")
print("=" * 70)

def detect_language(text):
    """Heuristic English detection optimized for financial text."""
    if not text or len(text) < 3:
        return False
    
    english_words = {
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
        'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
        'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
        'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
        'is', 'are', 'was', 'were', 'been', 'has', 'had', 'does', 'did', 'can',
        'could', 'should', 'stock', 'market', 'price', 'buy', 'sell', 'trading',
        'shares', 'earnings', 'revenue', 'profit', 'loss', 'quarter', 'reports',
        'hold', 'long', 'short', 'call', 'put', 'bull', 'bear', 'rally', 
        'surge', 'plunge', 'dip', 'gap', 'eps', 'guidance', 'forecast'
    }
    
    words = re.findall(r'\b[a-z]+\b', text.lower())
    
    if len(words) < 5:
        return any(word in english_words for word in words)
    
    english_count = sum(1 for word in words if word in english_words)
    return (english_count / len(words)) >= 0.3


def remove_source_prefix(text):
    """Remove source attribution patterns from news headlines."""
    text = re.sub(r'^(Update|Brief|Breaking|Alert|Exclusive|News|Report|Analysis|Video|Audio)\s*:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^[A-Za-z\s&]+\s*-\s*', '', text)
    
    return text


def remove_urls(text):
    """Remove URLs from text."""
    text = re.sub(r'http\S+|www\.\S+', '', text)
    return text


def remove_special_patterns(text):
    """Remove email addresses and excessive punctuation."""
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'([!?.]){2,}', r'\1', text)
    
    return text


def clean_whitespace(text):
    """Remove extra whitespace."""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def preprocess_text(text):
    """Apply text cleaning pipeline to financial news headlines."""
    if not text or not isinstance(text, str):
        return None
    
    text = html.unescape(text)
    text = remove_source_prefix(text)
    text = remove_urls(text)
    text = remove_special_patterns(text)
    text = re.sub(r'([!?.,();:])', r' \1 ', text)
    text = clean_whitespace(text)
    
    if len(text) < 10:
        return None
    
    words = text.split()
    if len(words) < 3:
        return None
        
    return text


def preprocess_news():
    """Execute preprocessing pipeline on news headlines corpus."""
    print("\n1. Connecting to MongoDB...")
    client = pymongo.MongoClient(MONGO_URI)
    db = client[DB_NAME]
    raw_collection = db[COLLECTION_NEWS_ARTICLES]
    cleaned_collection = db[COLLECTION_CLEANED_NEWS]
    
    print("   Clearing previous cleaned news...")
    cleaned_collection.delete_many({})
    print("   Ready for preprocessing.")
    
    print("\n2. Discovering tickers from database...")
    tickers = raw_collection.distinct('ticker')
    print(f"   Found {len(tickers)} tickers: {tickers}")
    
    if not tickers:
        print("   ERROR: No tickers found in database. Run data ingestion first.")
        client.close()
        return
    
    total_articles = raw_collection.count_documents({})
    print(f"   Total articles in database: {total_articles:,}")
    
    print(f"\n3. Preprocessing news headlines...")
    print("=" * 70)
    
    total_processed = 0
    total_filtered = 0
    total_cleaned = 0
    
    for ticker in tickers:
        print(f"\n   Processing {ticker}...")
        
        query = {'ticker': ticker}
        articles = list(raw_collection.find(query))
        
        if not articles:
            print(f"     No articles found for {ticker}")
            continue
        
        print(f"     Found {len(articles)} articles")
        
        filtered = 0
        cleaned = 0
        cleaned_docs = []
        batch_count = 0
        seen_titles = set()
        duplicates = 0
        
        for article in articles:
            total_processed += 1
            original_title = article.get('title', '')
            
            if original_title in seen_titles:
                duplicates += 1
                filtered += 1
                continue
            seen_titles.add(original_title)
            
            if not detect_language(original_title):
                filtered += 1
                continue
            
            cleaned_title = preprocess_text(original_title)
            
            if cleaned_title is None:
                filtered += 1
            else:
                cleaned_doc = {
                    'ticker': article['ticker'],
                    'platform': article.get('platform', 'News'),
                    'article_id': article.get('article_id', article.get('_id')),
                    'original_title': original_title,
                    'cleaned_title': cleaned_title,
                    'source': article.get('source', 'Unknown'),
                    'published_at': article['published_at']
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
        
        print(f"     Cleaned: {cleaned}, Filtered: {filtered} (Duplicates: {duplicates})")
    
    print(f"\n4. Summary:")
    print("=" * 70)
    print(f"   Total articles processed: {total_processed}")
    print(f"   Cleaned (English): {total_cleaned}")
    print(f"   Filtered out: {total_filtered}")
    print(f"   Success rate: {100 * total_cleaned / total_processed:.1f}%")
    print(f"\n   Preprocessing complete.")
    
    client.close()


if __name__ == "__main__":
    preprocess_news()
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
