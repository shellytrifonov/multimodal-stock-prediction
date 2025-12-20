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
    COLLECTION_CLEANED_NEWS, COLLECTION_NEWS_WITH_SENTIMENT
)

BATCH_SIZE = 32

print("=" * 70)
print("Sentiment Analysis with FinBERT")
print("=" * 70)

print("\nLoading FinBERT model...")
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Using device: {device}")

id2label = model.config.id2label
if id2label[0].lower() != 'positive':
    print(f"WARNING: FinBERT Label mismatch! Expected 'positive' at index 0, got '{id2label[0]}'")
    print(f"Full label mapping: {id2label}")
else:
    print(f"✓ Label mapping verified: {id2label}")

LABEL_MAPPING = {0: 'positive', 1: 'negative', 2: 'neutral'}


def calculate_sentiment_batch(texts):
    """Compute sentiment probabilities for batch of news headlines."""
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
        p_pos = prob_vector[0]
        p_neg = prob_vector[1]
        p_neu = prob_vector[2]
        polarity = p_pos - p_neg
        results.append((polarity, p_pos, p_neg, p_neu))
    
    return results


def process_sentiment_scores():
    """Execute sentiment analysis pipeline on preprocessed news headlines."""
    print(f"\n1. Connecting to MongoDB...")
    client = pymongo.MongoClient(MONGO_URI)
    db = client[DB_NAME]
    cleaned_collection = db[COLLECTION_CLEANED_NEWS]
    sentiment_collection = db[COLLECTION_NEWS_WITH_SENTIMENT]
    
    print("   Clearing previous sentiment data...")
    sentiment_collection.delete_many({})
    print("   Ready for sentiment analysis.")
    
    print(f"\n2. Discovering tickers from database...")
    tickers = cleaned_collection.distinct('ticker')
    print(f"   Found {len(tickers)} tickers: {tickers}")
    
    if not tickers:
        print("   ERROR: No tickers found in cleaned_news collection.")
        print("   Please run preprocess_news.py first.")
        client.close()
        return
    
    print(f"\n3. Collecting cleaned news headlines...")
    all_articles = list(cleaned_collection.find({}))
    total_to_process = len(all_articles)
    
    for ticker in tickers:
        count = cleaned_collection.count_documents({'ticker': ticker})
        print(f"   {ticker}: {count} cleaned headlines")
    
    print(f"\n   Total headlines to process: {total_to_process}")
    
    if total_to_process == 0:
        print("   ERROR: No headlines found for sentiment analysis!")
        print("   Please run preprocess_news.py first.")
        client.close()
        return
    
    print(f"\n4. Processing {total_to_process} headlines in batches of {BATCH_SIZE}...")
    print(f"   Total batches: {(total_to_process + BATCH_SIZE - 1) // BATCH_SIZE}")
    
    polarities = []
    all_sentiment_docs = []
    
    for batch_start in tqdm(range(0, len(all_articles), BATCH_SIZE), desc="Processing batches"):
        batch_end = min(batch_start + BATCH_SIZE, len(all_articles))
        batch_articles = all_articles[batch_start:batch_end]
        
        batch_texts = []
        valid_indices = []
        
        for idx, article in enumerate(batch_articles):
            text = article.get('cleaned_title', '')
            if text and str(text).strip() != '':
                batch_texts.append(str(text))
                valid_indices.append(idx)
            else:
                batch_texts.append("")
        
        if batch_texts and any(t for t in batch_texts):
            texts_to_process = [batch_texts[i] for i in valid_indices]
            
            if texts_to_process:
                batch_results = calculate_sentiment_batch(texts_to_process)
            else:
                batch_results = []
        else:
            batch_results = []
        
        result_idx = 0
        batch_sentiment_docs = []
        
        for article_idx, article in enumerate(batch_articles):
            if article_idx in valid_indices and result_idx < len(batch_results):
                polarity, p_pos, p_neg, p_neu = batch_results[result_idx]
                result_idx += 1
                
                polarities.append(polarity)
                
                sentiment_doc = {
                    'ticker': article['ticker'],
                    'platform': article.get('platform', 'News'),
                    'article_id': article.get('article_id'),
                    'original_title': article.get('original_title', ''),
                    'cleaned_title': article.get('cleaned_title', ''),
                    'source': article.get('source', 'Unknown'),
                    'published_at': article['published_at'],
                    'sentiment_score': float(polarity),
                    'p_pos': float(p_pos),
                    'p_neg': float(p_neg),
                    'p_neu': float(p_neu)
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
