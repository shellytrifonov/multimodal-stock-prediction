# Stock Price Direction Prediction Using Multimodal LSTM Fusion

Final graduation project implementing a hybrid deep learning system that predicts next-day stock price direction by fusing historical price data with Twitter sentiment analysis.

## Project Overview

This system combines two data sources to predict whether a stock will go up or down the next day:
- **Historical stock prices** (60-day sequences with technical indicators)
- **Twitter sentiment** (72-hour aggregated sentiment windows)

The architecture uses separate LSTM encoders for each modality, then fuses their outputs through a fully connected network for binary classification.

**Result:** 54.03% test accuracy (marginally above baseline), demonstrating that aggregated Twitter sentiment provides limited predictive value for next-day stock direction.

## Technology Stack

- **Deep Learning:** PyTorch 2.0+
- **NLP:** Hugging Face Transformers (RoBERTa sentiment model)
- **Data Processing:** pandas, NumPy
- **Database:** MongoDB
- **Visualization:** matplotlib, seaborn
- **Web Interface:** Streamlit

## Project Structure

```
stock_prediction_project/
├── config/
│   └── db_config.py              # MongoDB configuration
├── data/
│   ├── fetch_stock_data.py       # Download stock prices from Yahoo Finance
│   └── fetch_twitter_data.py     # Load Twitter dataset from Kaggle
├── pipelines/
│   ├── stock/
│   │   ├── preprocess_stock.py   # Clean and add technical indicators
│   │   └── build_stock_training_data.py  # Create 60-day sequences
│   └── twitter/
│       ├── preprocess_tweets.py  # Text cleaning
│       ├── twitter_sentiment.py  # RoBERTa sentiment scoring
│       ├── aggregate_hourly_twitter.py   # Hourly aggregation
│       └── build_twitter_training_data.py  # Create 72-hour sequences
├── models/
│   ├── stock_lstm.py             # Stock LSTM encoder
│   ├── twitter_lstm.py           # Twitter LSTM encoder
│   ├── hybrid_fusion_model.py    # Fusion network
│   ├── train_stock_lstm.py       # Train stock encoder
│   ├── train_twitter_lstm.py     # Train sentiment encoder
│   └── train_hybrid_fusion.py    # Train fusion model
├── scripts/
│   ├── run_stock_pipeline.py     # End-to-end stock pipeline
│   ├── run_twitter_pipeline.py   # End-to-end Twitter pipeline
│   └── run_hybrid_pipeline.py    # Full training pipeline
├── app.py                        # Streamlit web interface
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Setup Instructions

### Prerequisites

- Python 3.8+
- MongoDB installed and running locally
- GPU recommended (but not required)
- ~10GB disk space for datasets

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd stock_prediction_project
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure MongoDB**

Edit `config/db_config.py` to match your MongoDB setup:
```python
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "stock_prediction"
```

Start MongoDB if it's not running:
```bash
# On Linux/Mac
sudo systemctl start mongod

# On Windows
net start MongoDB
```

4. **Verify installation**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

## Usage Guide

### Full Pipeline (Recommended)

Run the complete training pipeline from scratch:

```bash
# 1. Stock pipeline: fetch data → preprocess → build sequences → train LSTM
python scripts/run_stock_pipeline.py

# 2. Twitter pipeline: fetch data → clean → sentiment → aggregate → train LSTM
python scripts/run_twitter_pipeline.py

# 3. Fusion: align datasets → train fusion model
python scripts/run_hybrid_pipeline.py

# 4. Evaluate results
python evaluate_model.py
```

**Expected runtime:**
- Stock pipeline: ~15 minutes
- Twitter pipeline: ~45 minutes (sentiment extraction is slow)
- Fusion training: ~10 minutes
- **Total: ~70 minutes** (with GPU)

### Step-by-Step Execution

If you prefer to run stages individually:

#### Stage 1: Data Collection

```bash
# Download stock prices (2015-2020) for 6 tickers
python data/fetch_stock_data.py

# Load Twitter dataset from Kaggle (requires Kaggle API key)
python data/fetch_twitter_data.py
```

#### Stage 2: Stock Preprocessing

```bash
# Clean stock data and add 19 technical indicators
python pipelines/stock/preprocess_stock.py

# Build 60-day sliding window sequences
python pipelines/stock/build_stock_training_data.py
```

#### Stage 3: Twitter Preprocessing

```bash
# Clean tweet text (remove URLs, emojis, etc.)
python pipelines/twitter/preprocess_tweets.py

# Extract sentiment using RoBERTa model
python pipelines/twitter/twitter_sentiment.py

# Aggregate to hourly features
python pipelines/twitter/aggregate_hourly_twitter.py

# Build 72-hour sliding window sequences
python pipelines/twitter/build_twitter_training_data.py
```

#### Stage 4: Model Training

```bash
# Train stock LSTM (100 epochs, ~10 min)
python models/train_stock_lstm.py

# Train Twitter LSTM (100 epochs, ~15 min)
python models/train_twitter_lstm.py

# Train fusion model with frozen encoders (50 epochs, ~5 min)
python models/train_hybrid_fusion.py
```

#### Stage 5: Evaluation

```bash
# Generate metrics, confusion matrix, and per-ticker analysis
python evaluate_model.py
```

### Web Interface

Launch the Streamlit dashboard to visualize predictions:

```bash
streamlit run app.py
```

The interface provides:
- Ticker selection
- Date range filtering
- Prediction vs actual comparison
- Confidence scores
- Historical price charts


## Troubleshooting

### Common Issues

**MongoDB Connection Error**
```
pymongo.errors.ServerSelectionTimeoutError
```
**Fix:** Start MongoDB service and verify connection string in `config/db_config.py`

**CUDA Out of Memory**
```
RuntimeError: CUDA out of memory
```
**Fix:** Reduce batch size in training scripts or use CPU (slower but works)

**Missing Kaggle Credentials**
```
OSError: Could not find kaggle.json
```
**Fix:** Download Kaggle API token from kaggle.com/account and place in `~/.kaggle/`

**Sentiment Extraction Slow**
```
Processing takes hours...
```
**Expected:** RoBERTa inference is slow on CPU. Use GPU or reduce dataset size for testing.

### Data Validation

Check that datasets were created correctly:

```python
# Verify stock data
python -c "import numpy as np; data = np.load('stock_lstm_training_data.npz'); print(data['X_train'].shape)"
# Expected: (5664, 60, 25) or similar

# Verify Twitter data
python -c "import numpy as np; data = np.load('twitter_lstm_training_data.npz'); print(data['X_train'].shape)"
# Expected: (5664, 72, 10) or similar
```

## Project Limitations

1. **Sentiment granularity:** 72-hour aggregation may be too coarse for intraday market reactions
2. **Data source:** General Twitter discourse lacks specificity of financial news
3. **Prediction target:** Next-day direction may not align with sentiment dynamics
4. **Class imbalance:** Not addressed with weighted loss during initial training
5. **Architecture:** Simple concatenation fusion may be suboptimal vs. attention-based fusion

## Future Improvements

- Try financial news headlines instead of general Twitter data
- Use FinBERT for domain-specific sentiment extraction
- Implement intraday prediction (next hour instead of next day)
- Add class-weighted loss or threshold tuning
- Explore cross-attention fusion mechanisms
- Incorporate portfolio-level forecasting across multiple stocks

## Dependencies

See `requirements.txt` for complete list. Key libraries:

- `torch>=2.0.0` - Deep learning framework
- `transformers>=4.30.0` - RoBERTa sentiment model
- `pandas>=2.0.0` - Data manipulation
- `pymongo>=4.6.0` - MongoDB driver
- `yfinance>=0.2.0` - Stock data API
- `scikit-learn>=1.3.0` - Metrics and scaling
- `streamlit>=1.28.0` - Web interface

## License

This project is submitted as a graduation requirement. All code is original work except where cited.

## Author

Created as a final year project demonstrating multimodal deep learning for financial forecasting.

## Acknowledgments

- **Datasets:** Kaggle Twitter dataset (omermetinn/tweets-about-the-top-companies-from-2015-to-2020), Yahoo Finance API
- **Models:** Cardiff NLP RoBERTa sentiment model (cardiffnlp/twitter-roberta-base-sentiment)
- **References:** See project documentation for academic citations
