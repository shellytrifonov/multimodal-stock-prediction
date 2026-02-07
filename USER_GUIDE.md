# User Guide: Stock Prediction System

This guide walks you through setting up and using the stock prediction system from scratch.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Detailed Setup](#detailed-setup)
3. [Running Your First Prediction](#running-your-first-prediction)
4. [Understanding the Output](#understanding-the-output)
5. [Troubleshooting](#troubleshooting)
6. [Advanced Usage](#advanced-usage)

---

## Quick Start

**For experienced users who just want to get started:**

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start MongoDB
# (depends on your OS - see detailed setup)

# 3. Run full pipeline
python scripts/run_stock_pipeline.py
python scripts/run_twitter_pipeline.py
python scripts/run_hybrid_pipeline.py

# 4. View results
python evaluate_model.py
streamlit run app.py
```

**Expected time:** ~70 minutes (with GPU)

---

## Detailed Setup

### Step 1: Install Python Dependencies

Make sure you have Python 3.8 or higher installed:

```bash
python --version
```

Install required packages:

```bash
pip install -r requirements.txt
```

**What this installs:**
- PyTorch for deep learning
- Transformers for sentiment analysis
- pandas/NumPy for data processing
- MongoDB driver
- Visualization libraries
- Streamlit for the web interface

### Step 2: Setup MongoDB

MongoDB stores all the raw and processed data.

**On Windows:**
1. Download MongoDB from mongodb.com
2. Install with default settings
3. Start MongoDB:
   ```bash
   net start MongoDB
   ```

**On Mac:**
```bash
brew install mongodb-community
brew services start mongodb-community
```

**On Linux:**
```bash
sudo systemctl start mongod
```

**Verify it's running:**
```bash
# This should connect without errors
python -c "import pymongo; pymongo.MongoClient('mongodb://localhost:27017/').server_info()"
```

### Step 3: Configure Database Connection

Open `config/db_config.py` and verify the settings:

```python
MONGO_URI = "mongodb://localhost:27017/"  # Default MongoDB address
DB_NAME = "stock_prediction"              # Database name
```

The tickers to analyze are also configured here:
```python
LIMIT_TICKERS = ['AAPL', 'AMZN', 'GOOG', 'GOOGL', 'MSFT', 'TSLA']
```

You can modify this list if you want to analyze different stocks.

### Step 4: (Optional) Setup Kaggle API

The Twitter dataset comes from Kaggle. You'll need API credentials:

1. Go to kaggle.com → Your Account → Create New API Token
2. Download `kaggle.json`
3. Place it in:
   - **Windows:** `C:\Users\<username>\.kaggle\kaggle.json`
   - **Linux/Mac:** `~/.kaggle/kaggle.json`

---

## Running Your First Prediction

### Option A: Automated Pipeline (Recommended)

Run the complete system with three commands:

**1. Stock Data Pipeline**
```bash
python scripts/run_stock_pipeline.py
```

This will:
- Download 6 years of stock prices from Yahoo Finance
- Calculate 19 technical indicators
- Create 60-day sliding window sequences
- Train the stock LSTM encoder

**Time:** ~15 minutes

**2. Twitter Data Pipeline**
```bash
python scripts/run_twitter_pipeline.py
```

This will:
- Load Twitter dataset from Kaggle
- Clean tweet text
- Extract sentiment using RoBERTa
- Aggregate to hourly features
- Create 72-hour sliding window sequences
- Train the Twitter LSTM encoder

**Time:** ~45 minutes (sentiment extraction is slow)

**3. Fusion Model Pipeline**
```bash
python scripts/run_hybrid_pipeline.py
```

This will:
- Align stock and Twitter data by date
- Train the fusion model
- Save the final trained model

**Time:** ~10 minutes

### Option B: Step-by-Step Execution

If you want more control or need to debug issues:

**Step 1: Collect Data**
```bash
python data/fetch_stock_data.py
python data/fetch_twitter_data.py
```

**Step 2: Preprocess Stock Data**
```bash
python pipelines/stock/preprocess_stock.py
python pipelines/stock/build_stock_training_data.py
```

**Step 3: Preprocess Twitter Data**
```bash
python pipelines/twitter/preprocess_tweets.py
python pipelines/twitter/twitter_sentiment.py
python pipelines/twitter/aggregate_hourly_twitter.py
python pipelines/twitter/build_twitter_training_data.py
```

**Step 4: Train Models**
```bash
python models/train_stock_lstm.py
python models/train_twitter_lstm.py
python models/train_hybrid_fusion.py
```

---

## Understanding the Output

### Training Output

During training, you'll see progress like this:

```
Epoch  10/100 | Train Loss: 0.6543 Acc: 58.2% | Test Loss: 0.6821 Acc: 53.1%
Epoch  20/100 | Train Loss: 0.6234 Acc: 62.4% | Test Loss: 0.6897 Acc: 52.8%
...
Best: 54.0% at epoch 37
```

**What this means:**
- **Train Acc:** How well the model fits the training data
- **Test Acc:** How well it generalizes to unseen data
- **Best:** The model saved the checkpoint with highest test accuracy

### Evaluation Results

Run `python evaluate_model.py` to see detailed metrics:

**Example output:**
```
Hybrid Fusion Model Evaluation
================================================================================

Overall Metrics:
   Accuracy:    54.03%
   Precision:   54.49%
   Recall:      97.68%
   Specificity: 1.09%
   F1-Score:    69.96%
   ROC-AUC:     0.4986

Confusion Matrix:
                Predicted Down    Predicted Up
Actual Down           7                633
Actual Up            20                756

Per-Ticker Accuracy:
   AAPL: 57.63%
   AMZN: 55.08%
   GOOG: 52.12%
   GOOGL: 54.24%
   MSFT: 58.90%
   TSLA: 47.46%
```

**Key metrics explained:**

- **Accuracy:** Overall correctness (54% is slightly better than random 50%)
- **Precision:** When it predicts "Up", how often is it correct?
- **Recall:** Of all actual "Up" movements, how many did it catch?
- **Specificity:** Of all actual "Down" movements, how many did it catch?
- **ROC-AUC:** Discrimination ability (0.50 = random, 1.0 = perfect)

**In this case:** The model predicts "Up" almost always (99.5%), which is why recall is very high but specificity is near zero.

### Saved Model Files

After training, you'll find these files:

```
models/trained/
├── stock_lstm_trained.pth        # Stock encoder weights
├── twitter_lstm_trained.pth      # Twitter encoder weights
└── hybrid_fusion_trained.pth     # Fusion network weights
```

You can load these for inference without retraining.

---

## Using the Web Interface

Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

Your browser will open at `http://localhost:8501`

**Features:**

1. **Ticker Selection:** Choose which stock to analyze
2. **Date Range:** Filter predictions by date
3. **Prediction Table:** See all predictions with confidence scores
4. **Charts:** Visualize predictions vs actual movements
5. **Performance Metrics:** Per-ticker accuracy breakdown

**Example workflow:**
1. Select "AAPL" from dropdown
2. Choose date range "2019-06-01" to "2019-12-31"
3. See all predictions for Apple in that period
4. Check accuracy and confidence scores

---

## Troubleshooting

### Problem: MongoDB won't start

**Symptoms:**
```
pymongo.errors.ServerSelectionTimeoutError
```

**Solutions:**
1. Check if MongoDB is running:
   ```bash
   # Windows
   sc query MongoDB
   
   # Linux/Mac
   systemctl status mongod
   ```

2. If not running, start it (see Step 2 in Detailed Setup)

3. Verify connection string in `config/db_config.py` matches your MongoDB setup

### Problem: CUDA out of memory

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size in training scripts:
   - `train_stock_lstm.py`: Change `batch_size = 64` to `batch_size = 32`
   - `train_twitter_lstm.py`: Already uses `batch_size = 8`

2. Use CPU instead (slower but works):
   - Training will automatically fall back to CPU if CUDA unavailable

### Problem: Sentiment extraction is too slow

**Symptoms:**
Stuck at "Processing batch X/Y..." for hours

**Solutions:**
1. **Use GPU:** Speeds up RoBERTa inference by 10-20x
2. **Reduce dataset:** Modify `config/db_config.py` to use fewer tickers
3. **Be patient:** ~100k tweets takes ~45 minutes on CPU

### Problem: Kaggle dataset download fails

**Symptoms:**
```
OSError: Could not find kaggle.json
```

**Solutions:**
1. Get API token from kaggle.com/account
2. Download `kaggle.json`
3. Place in correct location (see Step 4 in Detailed Setup)
4. Verify file permissions:
   ```bash
   chmod 600 ~/.kaggle/kaggle.json  # Linux/Mac only
   ```

### Problem: Training accuracy stays at ~50%

**This is expected behavior!** 

Stock prediction is inherently difficult. The results show that:
- Twitter sentiment alone doesn't provide strong signal
- 54% accuracy is only marginally better than random guessing
- This is a valuable negative result for research

### Problem: Missing training data files

**Symptoms:**
```
FileNotFoundError: stock_lstm_training_data.npz not found
```

**Solutions:**
1. You need to run preprocessing first:
   ```bash
   python pipelines/stock/build_stock_training_data.py
   ```
2. Check that earlier steps completed successfully
3. Files should be created in the project root directory

---

## Advanced Usage

### Training with Different Hyperparameters

Edit the training scripts to experiment:

**Stock LSTM (`models/train_stock_lstm.py`):**
```python
num_epochs = 100          # Try 150 or 200
batch_size = 64           # Try 32 or 128
learning_rate = 0.001     # Try 0.0005 or 0.002
```

**Twitter LSTM (`models/train_twitter_lstm.py`):**
```python
hidden_size = 64          # Try 32 or 128
dropout = 0.4             # Try 0.3 or 0.5
```

### Using Different Tickers

Modify `config/db_config.py`:

```python
LIMIT_TICKERS = ['NVDA', 'META', 'NFLX']  # Different stocks
```

Then re-run the entire pipeline from data fetching.

### Modifying Technical Indicators

In `pipelines/stock/preprocess_stock.py`, you can add/remove indicators:

```python
# Add your own technical indicators
df['custom_indicator'] = calculate_custom_indicator(df)
```

Update `num_features` accordingly in training scripts.

### Training on Different Date Ranges

The system automatically uses the full date range in the datasets (2015-2020).

To use a subset:
1. Filter data in `build_stock_training_data.py` and `build_twitter_training_data.py`
2. Add date filters before creating sequences

### Exporting Predictions

After evaluation, predictions are available in memory. To save:

```python
# In evaluate_model.py, add at the end:
results_df = pd.DataFrame({
    'date': aligned_dates,
    'ticker': aligned_tickers,
    'actual': y_test,
    'predicted': predictions,
    'probability': probabilities
})
results_df.to_csv('predictions.csv', index=False)
```

---

## Performance Benchmarks

**Hardware:** Google Colab (Tesla T4 GPU, 12GB RAM)

| Stage | Time | Output |
|-------|------|--------|
| Stock data fetch | 2 min | ~15k price records |
| Twitter data load | 5 min | ~100k tweets |
| Stock preprocessing | 3 min | Technical indicators added |
| Twitter cleaning | 10 min | Text normalized |
| Sentiment extraction | 30 min | RoBERTa inference |
| Hourly aggregation | 5 min | ~50k hourly features |
| Stock LSTM training | 10 min | 100 epochs |
| Twitter LSTM training | 15 min | 100 epochs |
| Fusion training | 5 min | 50 epochs |
| **Total** | **~85 min** | Trained models |

**On CPU:** Expect 2-3x longer, mainly for sentiment extraction.

---

## Data Storage Requirements

| Component | Size | Location |
|-----------|------|----------|
| MongoDB database | ~500 MB | `/data/db/` |
| Training sequences | ~200 MB | `.npz` files |
| Model checkpoints | ~50 MB | `models/trained/` |
| **Total** | **~750 MB** | - |

---

## Getting Help

If you encounter issues not covered here:

1. Check that all steps in "Detailed Setup" were completed
2. Verify MongoDB is running and accessible
3. Ensure you have sufficient disk space (~1GB free)
4. Try running individual scripts to isolate the problem
5. Check console output for specific error messages

**Common error patterns:**

- `ModuleNotFoundError` → Missing dependency, run `pip install -r requirements.txt`
- `FileNotFoundError` → Missing data file, run preprocessing steps
- `ConnectionError` → MongoDB not running or wrong connection string
- `CUDA error` → GPU issue, reduce batch size or use CPU

---

## Next Steps

After successfully running the system:

1. **Analyze Results:** Review evaluation metrics and confusion matrix
2. **Try Different Tickers:** Modify configuration and retrain
3. **Experiment with Hyperparameters:** Adjust learning rates, dropout, etc.
4. **Improve Architecture:** Try different fusion strategies or add more features
5. **Read Documentation:** See project report for methodology details

Good luck with your stock prediction experiments!
