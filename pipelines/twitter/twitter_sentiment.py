import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pymongo
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
from config.db_config import (
    MONGO_URI, DB_NAME,
    COLLECTION_CLEANED_TWEETS, COLLECTION_TWEETS_WITH_SENTIMENT,
    LIMIT_TICKERS, USE_SAMPLE, SAMPLE_SIZE_PER_TICKER
)

BATCH_SIZE = 32

print("=" * 70)
print("Sentiment Analysis with RoBERTa")
print("=" * 70)

print("\nLoading RoBERTa model...")
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Using device: {device}")

id2label = model.config.id2label
if id2label[0].lower() != 'negative':
    print(f"WARNING: RoBERTa Label mismatch! Expected 'negative' at index 0, got '{id2label[0]}'")
    print(f"Full label mapping: {id2label}")
else:
    print(f"✓ Label mapping verified: {id2label}")


def calculate_polarity_batch(texts):
    """Compute sentiment probabilities for batch of tweets."""
    if not texts:
        return []
    
    inputs = tokenizer(
        texts, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512, 
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
    
    results = []
    for prob_vector in probs:
        p_negative, p_neutral, p_positive = prob_vector[0], prob_vector[1], prob_vector[2]
        polarity = p_positive - p_negative
        results.append((polarity, p_negative, p_neutral, p_positive))
    
    return results


def process_sentiment_scores():
    """Execute sentiment analysis pipeline on preprocessed tweets."""
    print(f"\n1. Connecting to MongoDB...")
    client = pymongo.MongoClient(MONGO_URI)
    db = client[DB_NAME]
    cleaned_collection = db[COLLECTION_CLEANED_TWEETS]
    sentiment_collection = db[COLLECTION_TWEETS_WITH_SENTIMENT]
    
    print("   Clearing previous sentiment data...")
    sentiment_collection.delete_many({})
    print("   Ready for sentiment analysis.")
    
    print(f"\n2. Discovering tickers...")
    if LIMIT_TICKERS:
        tickers = LIMIT_TICKERS
        print(f"   Using specified tickers: {tickers}")
    else:
        tickers = cleaned_collection.distinct('ticker')
        print(f"   Found {len(tickers)} tickers in cleaned tweets: {tickers}")
    
    base_query = {}
    if LIMIT_TICKERS:
        base_query['ticker'] = {'$in': LIMIT_TICKERS}
    
    total_to_process = 0
    all_tweets = []
    
    print(f"\n3. Collecting cleaned tweets...")
    if USE_SAMPLE:
        for ticker in tickers:
            query = base_query.copy()
            query['ticker'] = ticker
            ticker_tweets = list(cleaned_collection.find(query).limit(SAMPLE_SIZE_PER_TICKER))
            all_tweets.extend(ticker_tweets)
            print(f"   {ticker}: {len(ticker_tweets)} cleaned tweets")
        total_to_process = len(all_tweets)
    else:
        all_tweets = list(cleaned_collection.find(base_query))
        total_to_process = len(all_tweets)
        print(f"   Total tweets to process: {total_to_process}")
    
    if total_to_process == 0:
        print("   All tweets already have sentiment scores!")
        client.close()
        return
    
    print(f"\n4. Processing {total_to_process} tweets in batches of {BATCH_SIZE}...")
    print(f"   Total batches: {(total_to_process + BATCH_SIZE - 1) // BATCH_SIZE}")
    
    polarities = []
    all_sentiment_docs = []
    
    for batch_start in tqdm(range(0, len(all_tweets), BATCH_SIZE), desc="Processing batches"):
        batch_end = min(batch_start + BATCH_SIZE, len(all_tweets))
        batch_tweets = all_tweets[batch_start:batch_end]
        
        batch_texts = []
        valid_indices = []
        
        for idx, tweet in enumerate(batch_tweets):
            text = tweet.get('cleaned_text', '')
            if text and str(text).strip() != '':
                batch_texts.append(str(text))
                valid_indices.append(idx)
            else:
                batch_texts.append("")
        
        if batch_texts and any(t for t in batch_texts):
            texts_to_process = [batch_texts[i] for i in valid_indices]
            
            if texts_to_process:
                batch_results = calculate_polarity_batch(texts_to_process)
            else:
                batch_results = []
        else:
            batch_results = []
        
        result_idx = 0
        batch_sentiment_docs = []
        
        for tweet_idx, tweet in enumerate(batch_tweets):
            if tweet_idx in valid_indices and result_idx < len(batch_results):
                polarity, p_neg, p_neu, p_pos = batch_results[result_idx]
                result_idx += 1
                
                polarities.append(polarity)
                
                sentiment_doc = {
                    'ticker': tweet['ticker'],
                    'platform': tweet.get('platform', 'Twitter'),
                    'post_id': tweet['post_id'],
                    'original_text': tweet.get('original_text', ''),
                    'cleaned_text': tweet.get('cleaned_text', ''),
                    'author': tweet.get('author'),
                    'created_at': tweet['created_at'],
                    'likes_count': tweet.get('likes_count', 0),
                    'retweet_count': tweet.get('retweet_count', 0),
                    'comment_count': tweet.get('comment_count', 0),
                    'sentiment_score': float(polarity),
                    'p_negative': float(p_neg),
                    'p_neutral': float(p_neu),
                    'p_positive': float(p_pos)
                }
                batch_sentiment_docs.append(sentiment_doc)
        if batch_sentiment_docs:
            sentiment_collection.insert_many(batch_sentiment_docs)
            all_sentiment_docs.extend(batch_sentiment_docs)
    
    print(f"\n5. Summary:")
    print("=" * 70)
    print(f"   Total processed: {len(all_sentiment_docs)}")
    print(f"   Avg Polarity: {np.mean(polarities):.4f}")
    print(f"   Positive (>0.1): {sum(p > 0.1 for p in polarities)}")
    print(f"   Neutral: {sum(-0.1 <= p <= 0.1 for p in polarities)}")
    print(f"   Negative (<-0.1): {sum(p < -0.1 for p in polarities)}")
    print(f"\n   Sentiment analysis complete.")
    
    client.close()


if __name__ == "__main__":
    process_sentiment_scores()
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
