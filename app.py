import os
import sys
import time
import subprocess
import streamlit as st
import pymongo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from models.stock_lstm import StockLSTM
from models.twitter_lstm import TwitterLSTM
from models.hybrid_fusion_model import HybridFusionModel

# =========================
# CONFIG
# =========================
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "StockPredictionDB"
COLL_TWEETS = "raw_tweets"
COLL_STOCKS = "stock_prices"
COLL_HOURLY_TW = "hourly_sentiment_tweets"   # expected fields: ticker, hour (ISODate), sentiment_score/avg

FETCH_TWITTER_CMD = [sys.executable, "data/fetch_twitter_data.py"]
FETCH_STOCK_CMD   = [sys.executable, "data/fetch_stock_data.py"]

TRAIN_STOCK_CMD   = [sys.executable, "scripts/run_stock_pipeline.py"]
TRAIN_TWITTER_CMD = [sys.executable, "scripts/run_twitter_pipeline.py"]
TRAIN_HYBRID_CMD  = [sys.executable, "scripts/run_hybrid_pipeline.py"]

MODEL_FILES = {
    "Stock LSTM": "models/trained/stock_lstm_trained.pth",
    "Twitter LSTM": "models/trained/twitter_lstm_trained.pth",
    "Hybrid Fusion": "models/trained/hybrid_fusion_trained.pth",
}

st.set_page_config(page_title="Hybrid Stock Prediction — Setup", layout="wide")

# =========================
# Helpers
# =========================
def get_db():
    return pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)[DB_NAME]

def get_stats():
    db = get_db()
    tweets_count = db[COLL_TWEETS].count_documents({})
    stock_count = db[COLL_STOCKS].count_documents({})
    tweets_sample = db[COLL_TWEETS].find_one({}, {"ticker": 1, "created_at": 1})
    stock_sample = db[COLL_STOCKS].find_one({}, {"ticker": 1, "date": 1})
    return tweets_count, stock_count, tweets_sample, stock_sample

def clear_data():
    db = get_db()
    db[COLL_TWEETS].delete_many({})
    db[COLL_STOCKS].delete_many({})

def run_cmd(cmd, log_box):
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines = []
    while True:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        if line:
            lines.append(line.rstrip())
            if len(lines) > 300:
                lines = lines[-300:]
            log_box.code("\n".join(lines))
    return proc.wait()

def ingest_all(log_box, status_box):
    status_box.info("Running Twitter ingestion...")
    rc1 = run_cmd(FETCH_TWITTER_CMD, log_box)
    if rc1 != 0:
        status_box.error(f"Twitter ingestion failed (exit {rc1}). Check logs.")
        return False

    status_box.info("Running Stock price fetch...")
    rc2 = run_cmd(FETCH_STOCK_CMD, log_box)
    if rc2 != 0:
        status_box.error(f"Stock fetch failed (exit {rc2}). Check logs.")
        return False

    status_box.success("Data ingestion completed ✅")
    return True

def file_status(path: str):
    if not os.path.exists(path):
        return False, None
    mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path)))
    size_mb = os.path.getsize(path) / (1024 * 1024)
    return True, (mtime, size_mb)

def delete_models():
    deleted = []
    for _, path in MODEL_FILES.items():
        if os.path.exists(path):
            os.remove(path)
            deleted.append(path)
    return deleted

def train_all(status_box, log_box, progress):
    steps = [
        ("Training Stock LSTM...", TRAIN_STOCK_CMD),
        ("Training Twitter LSTM...", TRAIN_TWITTER_CMD),
        ("Training Hybrid Fusion...", TRAIN_HYBRID_CMD),
    ]
    total = len(steps)
    for i, (label, cmd) in enumerate(steps, start=1):
        status_box.info(label)
        rc = run_cmd(cmd, log_box)
        if rc != 0:
            status_box.error(f"Step failed (exit {rc}). See logs above.")
            return False
        progress.progress(i / total)
    status_box.success("Training completed ✅")
    return True

# =========================
# Navigation state
# =========================
if "step" not in st.session_state:
    st.session_state.step = 1  # start at Step 1

# Top header
st.title("Hybrid Stock Prediction — Setup Wizard")
st.caption("Step-by-step: 1) Load Data → 2) Train Models → 3) Predict (next)")

# =========================
# STEP 1 — Data Setup
# =========================
def render_step_1():
    st.header("Step 1 — Data Setup")
    st.caption("Load Twitter + Stock data into MongoDB (one-time).")

    status_box = st.container()
    log_box = st.empty()

    try:
        tweets_count, stock_count, tweets_sample, stock_sample = get_stats()
    except Exception as e:
        status_box.error(f"MongoDB connection failed: {e}")
        return

    has_data = (tweets_count > 0) and (stock_count > 0)

    if has_data:
        st.success("Data is already loaded ✅ You may skip, or reset & re-ingest.")
    else:
        st.warning("No data found. Please fetch data to continue.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("raw_tweets count", f"{tweets_count:,}")
    c2.metric("stock_prices count", f"{stock_count:,}")
    c3.write("raw_tweets sample:")
    c3.json(tweets_sample or {})
    c4.write("stock_prices sample:")
    c4.json(stock_sample or {})

    st.divider()

    left, right = st.columns([2, 1])

    with left:
        if not has_data:
            if st.button("Fetch Data", type="primary"):
                ok = ingest_all(log_box, status_box)
                if ok:
                    time.sleep(0.3)
                    st.rerun()
        else:
            st.info("Data exists. You can continue to Step 2.")
            # Optional: still allow manual re-ingest without reset
            _ = st.checkbox("Skip data ingestion (already loaded)", value=True)

    with right:
        st.caption("Advanced")
        reset_clicked = st.button("Reset & Re-ingest Data", disabled=not has_data)
        if reset_clicked:
            st.warning("This will delete existing MongoDB data (raw_tweets + stock_prices).")
            confirm = st.checkbox("I understand, delete and re-ingest", value=False)
            if confirm and st.button("Confirm Reset & Re-ingest", type="primary"):
                status_box.info("Clearing existing data...")
                clear_data()
                status_box.info("Data cleared. Starting ingestion...")
                ok = ingest_all(log_box, status_box)
                if ok:
                    time.sleep(0.3)
                    st.rerun()

    st.divider()

    # NEXT button (only enabled if data exists)
    st.subheader("Next")
    if st.button("Next → Step 2 (Train Models)", type="primary", disabled=not has_data):
        st.session_state.step = 2
        st.rerun()

# =========================
# STEP 2 — Train Models
# =========================
def render_step_2():
    st.header("Step 2 — Train Models")
    st.caption("Train Stock LSTM, Twitter LSTM, then Hybrid Fusion. One-time for the dataset.")

    status_box = st.container()
    log_box = st.empty()

    # Show model file statuses
    cols = st.columns(3)
    all_ready = True
    for i, (name, path) in enumerate(MODEL_FILES.items()):
        ok, meta = file_status(path)
        if not ok:
            all_ready = False
        with cols[i]:
            if ok:
                mtime, size_mb = meta
                st.success(f"{name} ✅")
                st.caption(path)
                st.caption(f"Updated: {mtime}")
                st.caption(f"Size: {size_mb:.2f} MB")
            else:
                st.error(f"{name} ❌")
                st.caption(path)
                st.caption("Not found")

    st.divider()

    left, right = st.columns([2, 1])
    with left:
        if all_ready:
            st.success("All trained model files are present. You can skip training.")
            skip_training = st.checkbox("Skip training (models already trained)", value=True)
        else:
            st.warning("Some trained model files are missing. Training is required.")
            skip_training = st.checkbox("Skip training", value=False, disabled=True)

    with right:
        force_retrain = st.checkbox("Force retrain (delete existing .pth first)", value=False)

    progress = st.progress(0.0)

    # Train button
    if st.button("Train Models", type="primary", disabled=all_ready and skip_training):
        if force_retrain:
            deleted = delete_models()
            if deleted:
                status_box.info("Deleted existing model files:\n" + "\n".join(deleted))
            else:
                status_box.info("No existing model files to delete.")
        ok = train_all(status_box, log_box, progress)
        if ok:
            time.sleep(0.3)
            st.rerun()

    st.divider()

    nav_left, nav_right = st.columns([1, 1])
    with nav_left:
        if st.button("← Back to Step 1"):
            st.session_state.step = 1
            st.rerun()

    with nav_right:
        # We'll wire Step 3 next
        if st.button("Next → Step 3 (Predict)", type="primary", disabled=not all_ready):
            st.session_state.step = 3
            st.rerun()

# =========================
# STEP 3
# =========================
@st.cache_data
def load_npz_split(set_name: str = "test"):
    import numpy as np
    import pandas as pd

    def load_one(path):
        d = np.load(path, allow_pickle=True)
        X = d[f"X_{set_name}"]
        dates = pd.to_datetime(d[f"dates_{set_name}"])
        tickers = d[f"tickers_{set_name}"]
        # normalize tickers to str
        tickers = np.array([t.decode("utf-8") if isinstance(t, (bytes, np.bytes_)) else str(t) for t in tickers])
        return {"X": X, "dates": dates, "tickers": tickers}

    stock = load_one("stock_lstm_training_data.npz")
    twitter = load_one("twitter_lstm_training_data.npz")
    return stock, twitter

def build_alignment(stock, twitter):
    import numpy as np
    import pandas as pd

    stock_df = pd.DataFrame({
        "date": stock["dates"],
        "ticker": stock["tickers"],
        "stock_idx": np.arange(len(stock["dates"]))
    })
    twitter_df = pd.DataFrame({
        "date": twitter["dates"],
        "ticker": twitter["tickers"],
        "twitter_idx": np.arange(len(twitter["dates"]))
    })
    aligned = stock_df.merge(twitter_df, on=["date","ticker"], how="inner")
    return aligned.sort_values(["ticker","date"]).reset_index(drop=True)

@st.cache_resource
def load_models_from_npz(stock_feat_dim: int, twitter_feat_dim: int):
    import torch

    def infer_hidden_and_layers(sd, prefix):
        # prefix will be "lstm.lstm."
        w0 = sd[prefix + "weight_ih_l0"]
        hidden = w0.shape[0] // 4
        layers = 0
        while (prefix + f"weight_ih_l{layers}") in sd:
            layers += 1
        return hidden, layers

    def load_lstm_block(model, ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu")  # OrderedDict
        # Build a state_dict matching model.lstm.* (drop the leading "lstm.")
        # ckpt: "lstm.lstm.weight_ih_l0" -> model expects "lstm.weight_ih_l0" OR "lstm.lstm.weight_ih_l0"?
        # In your models, it's almost surely model.lstm.lstm.* and model.lstm.attention.* (same as ckpt but without leading "lstm.")
        filtered = {}
        for k, v in sd.items():
            if k.startswith("lstm."):
                filtered[k[len("lstm."):]] = v  # strip first "lstm."
        model.load_state_dict(filtered, strict=False)  # strict=False ignores classifier.*

        return sd

    # ---- STOCK ----
    stock_sd = torch.load(MODEL_FILES["Stock LSTM"], map_location="cpu")
    stock_hidden, stock_layers = infer_hidden_and_layers(stock_sd, prefix="lstm.lstm.")

    stock_lstm = StockLSTM(
        input_size=stock_feat_dim,
        hidden_size=stock_hidden,
        num_layers=stock_layers,
        dropout=0.2
    )
    load_lstm_block(stock_lstm, MODEL_FILES["Stock LSTM"])

    # ---- TWITTER ----
    tw_sd = torch.load(MODEL_FILES["Twitter LSTM"], map_location="cpu")
    tw_hidden, tw_layers = infer_hidden_and_layers(tw_sd, prefix="lstm.lstm.")

    twitter_lstm = TwitterLSTM(
        input_size=twitter_feat_dim,
        hidden_size=tw_hidden,
        num_layers=tw_layers,
        dropout=0.2
    )
    load_lstm_block(twitter_lstm, MODEL_FILES["Twitter LSTM"])

    # ---- FUSION ----
    fusion = HybridFusionModel(dropout_rate=0.2)
    fusion.load_state_dict(torch.load(MODEL_FILES["Hybrid Fusion"], map_location="cpu"))

    stock_lstm.eval()
    twitter_lstm.eval()
    fusion.eval()
    return stock_lstm, twitter_lstm, fusion

def fetch_stock_chart_30d(db, ticker: str, end_date: pd.Timestamp):
    # stock_prices.date stored as "YYYY-MM-DD"
    end_str = end_date.strftime("%Y-%m-%d")

    docs = list(db[COLL_STOCKS].find(
        {"ticker": ticker, "date": {"$lte": end_str}},
        {"_id": 0, "date": 1, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
        sort=[("date", -1)],
        limit=60  # fetch more then take 30 (safe)
    ))
    if not docs:
        return None

    docs.reverse()
    df = pd.DataFrame(docs)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").tail(30)
    return df

def get_available_tickers():
    db = get_db()
    return sorted(db[COLL_STOCKS].distinct("ticker"))

def find_last_trading_day(db, ticker, selected_date):
    # selected_date is a python date
    # stock 'date' stored as "YYYY-MM-DD" string
    target = selected_date.strftime("%Y-%m-%d")
    doc = db[COLL_STOCKS].find_one(
        {"ticker": ticker, "date": {"$lte": target}},
        sort=[("date", -1)]
    )
    return doc  # contains the actual date used

def load_stock_window(db, ticker, anchor_date_str, lookback_days=60):
    # Fetch last N trading days up to anchor_date_str
    cursor = db[COLL_STOCKS].find(
        {"ticker": ticker, "date": {"$lte": anchor_date_str}},
        {"_id": 0, "date": 1, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1, "adj_close": 1},
        sort=[("date", -1)],
        limit=lookback_days
    )
    rows = list(cursor)
    rows.reverse()  # oldest -> newest
    if len(rows) < lookback_days:
        return None, None
    df = pd.DataFrame(rows)
    X = df[["open","high","low","close","volume","adj_close"]].astype(float).to_numpy()
    return df, X

def load_twitter_window(db, ticker, anchor_date, hours=72):
    """
    Expects hourly_sentiment_tweets documents with:
      - ticker
      - hour: ISODate (datetime)
      - features: sentiment + maybe counts
    We'll build an 8-dim vector per hour. If your schema differs, tell me and I'll adapt.
    """
    end_dt = pd.Timestamp(anchor_date).tz_localize("UTC") + pd.Timedelta(hours=23)
    start_dt = end_dt - pd.Timedelta(hours=hours-1)

    cursor = db[COLL_HOURLY_TW].find(
        {"ticker": ticker, "hour": {"$gte": start_dt.to_pydatetime(), "$lte": end_dt.to_pydatetime()}},
        {"_id": 0},
        sort=[("hour", 1)]
    )
    rows = list(cursor)
    if len(rows) == 0:
        return None, None

    df = pd.DataFrame(rows)
    if "hour" not in df.columns:
        return None, None

    # Ensure continuous hourly grid
    df["hour"] = pd.to_datetime(df["hour"], utc=True)
    full_hours = pd.date_range(start=start_dt, end=end_dt, freq="H", tz="UTC")
    df = df.set_index("hour").reindex(full_hours)

    # Fill missing with 0 (or ffill). choose 0 to be safe.
    # You can adjust based on your pipeline semantics.
    df = df.fillna(0.0)

    # Build 8 features. We'll try to map common fields; missing -> 0.
    # Adjust here if your hourly docs have different field names.
    def col(name, default=0.0):
        return df[name].to_numpy() if name in df.columns else np.full(len(df), default, dtype=float)

    # candidate fields (adaptable)
    avg_sent = col("avg_sentiment", 0.0)
    sent     = col("sentiment_score", 0.0)   # sometimes exists instead of avg
    count    = col("count", 0.0)
    likes    = col("likes_sum", 0.0)
    rts      = col("retweets_sum", 0.0)
    replies  = col("replies_sum", 0.0)

    # Construct final per-hour vector (72, 8)
    X = np.stack([
        np.where(avg_sent != 0.0, avg_sent, sent),
        count,
        likes,
        rts,
        replies,
        np.zeros(len(df)),
        np.zeros(len(df)),
        np.zeros(len(df)),
    ], axis=1).astype(float)

    if X.shape[0] < hours:
        return None, None
    return df.reset_index().rename(columns={"index": "hour"}), X

def predict_one(stock_lstm, twitter_lstm, fusion, stock_X, twitter_X):
    stock_x = torch.tensor(stock_X, dtype=torch.float32).unsqueeze(0)    # (1,60,6)
    tw_x    = torch.tensor(twitter_X, dtype=torch.float32).unsqueeze(0)  # (1,72,8)
    with torch.no_grad():
        stock_feat = stock_lstm(stock_x)
        tw_feat = twitter_lstm(tw_x)
        prob = fusion(stock_feat, tw_feat).item()  # assumes fusion forward(stock, twitter)
    return prob
def plot_actual_vs_pred(dates, actual_close, pred_close):
    fig = plt.figure()
    plt.plot(dates, actual_close, label="Actual Close", color="blue")
    plt.plot(dates, pred_close, label="Predicted Close", color="red")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Actual vs Predicted (Projected) Close Price")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

def render_step_3():
    st.header("Step 3 — Predict")
    st.caption("Select a ticker + date. Inputs to the model are taken from the prepared NPZ (same as training).")

    # 1) check model files
    missing = [p for p in MODEL_FILES.values() if not os.path.exists(p)]
    if missing:
        st.error("Missing trained model files. Go back to Step 2.")
        st.code("\n".join(missing))
        return

    # 2) load NPZ + align
    try:
        use_set = st.radio("Dataset split", ["test", "train"], horizontal=True, index=0)
        stock, twitter = load_npz_split(use_set)
        aligned = build_alignment(stock, twitter)
    except Exception as e:
        st.error(str(e))
        return

    if aligned.empty:
        st.error("No aligned samples between Stock and Twitter NPZ files.")
        return

    # feature dims from NPZ (solves your mismatch)
    stock_feat_dim = stock["X"].shape[2]
    twitter_feat_dim = twitter["X"].shape[2]

    # 3) UI inputs
    tickers = sorted(aligned["ticker"].unique())
    col1, col2 = st.columns([1, 1])

    with col1:
        ticker = st.selectbox("Ticker", tickers)

    # available dates for this ticker (prediction dates)
    ticker_rows = aligned[aligned["ticker"] == ticker].copy()
    available_dates = sorted(ticker_rows["date"].dt.date.unique())

    with col2:
        default_date = available_dates[-1]
        picked = st.date_input("Date", value=default_date)

    # 4) pick exact or closest previous
    exact = ticker_rows[ticker_rows["date"].dt.date == picked]
    if exact.empty:
        prev = ticker_rows[ticker_rows["date"].dt.date < picked].tail(1)
        if prev.empty:
            st.error("No sample exists on or before this date for this ticker.")
            return
        row = prev.iloc[0]
        used_date = row["date"].date()
        st.info(f"No exact match for {picked}. Using closest previous date: {used_date}")
    else:
        row = exact.iloc[-1]
        used_date = row["date"].date()

    # 5) Show stock info + chart (Mongo)
    db = get_db()
    st.subheader("Stock info (MongoDB)")
    df30 = fetch_stock_chart_30d(db, ticker, pd.to_datetime(used_date))
    if df30 is None or df30.empty:
        st.warning("No stock_prices data found in MongoDB for chart/info.")
    else:
        last = df30.iloc[-1]
        a, b, c, d, e = st.columns(5)
        a.metric("Date used", str(used_date))
        b.metric("Close", f"{float(last['close']):.2f}")
        c.metric("Volume", f"{int(float(last['volume'])):,}")
        d.metric("Open", f"{float(last['open']):.2f}")
        change = (float(last["close"]) - float(last["open"])) / float(last["open"]) * 100.0 if float(last["open"]) else 0
        e.metric("Day change", f"{change:.2f}%")

    # 6) Load models once (cached)
    try:
        stock_lstm, twitter_lstm, fusion = load_models_from_npz(stock_feat_dim, twitter_feat_dim)
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return

    # 7) Build Actual vs Predicted price graph for last 30 aligned days
    st.subheader("Actual (Blue) vs Predicted (Red) — Last 30 Trading Days")

    # Use last 30 aligned samples up to used_date
    hist_rows = ticker_rows[ticker_rows["date"].dt.date <= used_date].tail(30).copy()
    if hist_rows.empty or len(hist_rows) < 5:
        st.warning("Not enough aligned history to plot actual vs predicted.")
    else:
        # Pull actual close prices from Mongo for these dates
        start_str = pd.to_datetime(hist_rows["date"].min()).date().strftime("%Y-%m-%d")
        end_str = pd.to_datetime(hist_rows["date"].max()).date().strftime("%Y-%m-%d")

        mongo_docs = list(db[COLL_STOCKS].find(
            {"ticker": ticker, "date": {"$gte": start_str, "$lte": end_str}},
            {"_id": 0, "date": 1, "close": 1},
            sort=[("date", 1)]
        ))

        if not mongo_docs:
            st.warning("No MongoDB stock_prices found for the selected range.")
        else:
            price_df = pd.DataFrame(mongo_docs)
            price_df["date"] = pd.to_datetime(price_df["date"])
            price_df = price_df.sort_values("date")

            hist_rows2 = hist_rows.copy()
            hist_rows2["date"] = pd.to_datetime(hist_rows2["date"])

            merged = hist_rows2.merge(price_df, on="date", how="inner").sort_values("date").reset_index(drop=True)

            if len(merged) < 5:
                st.warning("Not enough overlapping dates between NPZ samples and Mongo prices to plot.")
            else:
                # Compute p_up for each day (using NPZ inputs for that day)
                probs = []
                for _, r in merged.iterrows():
                    stock_X = stock["X"][int(r["stock_idx"])]
                    twitter_X = twitter["X"][int(r["twitter_idx"])]

                    stock_x = torch.tensor(stock_X, dtype=torch.float32).unsqueeze(0)
                    tw_x = torch.tensor(twitter_X, dtype=torch.float32).unsqueeze(0)

                    with torch.no_grad():
                        s_feat = stock_lstm(stock_x)
                        t_feat = twitter_lstm(tw_x)
                        try:
                            p_up = fusion(s_feat, t_feat).item()
                        except TypeError:
                            st.error("FusionModel signature mismatch. Tell me your forward(...) args and I’ll adapt.")
                            return
                    probs.append(float(p_up))

                merged["p_up"] = probs

                # Actual close series
                close = merged["close"].astype(float).to_numpy()

                # Typical daily move size in this window (avg abs return)
                rets = np.diff(close) / close[:-1]
                avg_move = float(np.nanmean(np.abs(rets))) if len(rets) > 0 else 0.01
                if not np.isfinite(avg_move) or avg_move <= 0:
                    avg_move = 0.01  # fallback 1%

                # Build projected predicted close curve (one-step projection)
                pred = np.zeros_like(close)
                pred[0] = close[0]
                for i in range(1, len(close)):
                    p = merged.loc[i - 1, "p_up"]
                    direction = 1.0 if p >= 0.5 else -1.0
                    confidence = abs(p - 0.5) * 2.0  # 0..1
                    move = avg_move * confidence * direction
                    pred[i] = close[i - 1] * (1.0 + move)

                # Plot with matplotlib (explicit colors)
                dates = merged["date"].dt.date.to_list()
                fig = plt.figure(figsize=(6,3))
                plt.plot(dates, close, label="Actual Close", color="blue")
                plt.plot(dates, pred, label="Predicted Close", color="red")
                plt.xlabel("Date")
                plt.ylabel("Price")
                plt.title("Actual vs Predicted (Projected) Close Price")
                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)

                st.caption(
                    "Blue = real close price from MongoDB. Red = projected close based on the model’s "
                    "direction + confidence (the model predicts direction, not an exact price)."
                )

    # 8) Single-day prediction for the selected date (same as before)
    st.subheader("Model prediction (Fusion) — Selected date")

    stock_X = stock["X"][int(row["stock_idx"])]
    twitter_X = twitter["X"][int(row["twitter_idx"])]

    stock_x = torch.tensor(stock_X, dtype=torch.float32).unsqueeze(0)
    tw_x = torch.tensor(twitter_X, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        stock_feat = stock_lstm(stock_x)
        tw_feat = twitter_lstm(tw_x)
        try:
            prob_up = fusion(stock_feat, tw_feat).item()
        except TypeError:
            st.error("FusionModel signature mismatch. Tell me your forward(...) args and I’ll adapt.")
            return

    trend = "Up" if prob_up >= 0.5 else "Down"
    conf = max(prob_up, 1 - prob_up)
    x1, x2, x3 = st.columns(3)
    x1.metric("Predicted Trend", trend)
    x2.metric("Up Probability", f"{prob_up:.3f}")
    x3.metric("Confidence", f"{conf*100:.1f}%")

    st.divider()
    if st.button("← Back to Step 2"):
        st.session_state.step = 2
        st.rerun()

# =========================
# Render
# =========================
if st.session_state.step == 1:
    render_step_1()
elif st.session_state.step == 2:
    render_step_2()
else:
    render_step_3()
