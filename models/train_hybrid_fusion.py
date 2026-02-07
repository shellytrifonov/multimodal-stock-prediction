import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from models.stock_lstm import StockLSTM
from models.twitter_lstm import TwitterLSTM
from models.hybrid_fusion_model import HybridFusionModel
import os

print("=" * 70)
print("Hybrid Fusion Model Training")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

print("\n1. Loading modality datasets...")

stock_path = 'stock_lstm_training_data.npz' if os.path.exists('stock_lstm_training_data.npz') else '../stock_lstm_training_data.npz'
if not os.path.exists(stock_path):
    print(f"   ERROR: Stock data not found at {stock_path}")
    exit(1)

stock_data = np.load(stock_path, allow_pickle=True)

# Combine train + test for alignment
stock_X_full = np.concatenate([stock_data["X_train"], stock_data["X_test"]], axis=0)
stock_y_full = np.concatenate([stock_data["y_train"], stock_data["y_test"]], axis=0)
stock_dates_full = np.concatenate([stock_data["dates_train"], stock_data["dates_test"]], axis=0)

twitter_path = 'twitter_lstm_training_data.npz' if os.path.exists('twitter_lstm_training_data.npz') else '../twitter_lstm_training_data.npz'
if not os.path.exists(twitter_path):
    print(f"   ERROR: Twitter data not found at {twitter_path}")
    exit(1)

twitter_data = np.load(twitter_path, allow_pickle=True)

twitter_X_full = np.concatenate([twitter_data["X_train"], twitter_data["X_test"]], axis=0)
twitter_y_full = np.concatenate([twitter_data["y_train"], twitter_data["y_test"]], axis=0)
twitter_dates_full = np.concatenate([twitter_data["dates_train"], twitter_data["dates_test"]], axis=0)

stock_num_features = stock_X_full.shape[2]
twitter_num_features = twitter_X_full.shape[2]
print(f"   Stock: {len(stock_X_full)} samples | {stock_X_full.shape}")
print(f"   Twitter: {len(twitter_X_full)} samples | {twitter_X_full.shape}")
print(f"   Features: Stock={stock_num_features}/day, Twitter={twitter_num_features}/hour")

print("\n2. Aligning samples by (date, ticker)...")

stock_dates_pd = pd.to_datetime(stock_dates_full)
twitter_dates_pd = pd.to_datetime(twitter_dates_full)

# Load tickers if available
if "tickers_train" in stock_data and "tickers_test" in stock_data:
    stock_tickers_full = np.concatenate([stock_data["tickers_train"], stock_data["tickers_test"]], axis=0)
else:
    stock_tickers_full = np.array(["UNKNOWN"] * len(stock_dates_pd))

if "tickers_train" in twitter_data and "tickers_test" in twitter_data:
    twitter_tickers_full = np.concatenate([twitter_data["tickers_train"], twitter_data["tickers_test"]], axis=0)
else:
    twitter_tickers_full = np.array(["UNKNOWN"] * len(twitter_dates_pd))

stock_df = pd.DataFrame({
    "date": stock_dates_pd,
    "ticker": stock_tickers_full,
    "s_idx": np.arange(len(stock_dates_pd))
})

twitter_df = pd.DataFrame({
    "date": twitter_dates_pd,
    "ticker": twitter_tickers_full,
    "t_idx": np.arange(len(twitter_dates_pd))
})

# Merge on BOTH date + ticker
merged = stock_df.merge(twitter_df, on=["date", "ticker"], how="inner")
merged = merged.sort_values(["date", "ticker"]).reset_index(drop=True)

print(f"   Stock: {len(stock_df)} | Twitter: {len(twitter_df)} | Aligned: {len(merged)}")

if len(merged) == 0:
    raise RuntimeError("No overlapping (date+ticker) pairs found between datasets!")

s_idx = merged["s_idx"].values
t_idx = merged["t_idx"].values
aligned_dates = merged["date"].values
aligned_tickers = merged["ticker"].values

aligned_stock_X = stock_X_full[s_idx]
aligned_stock_y = stock_y_full[s_idx]
aligned_twitter_X = twitter_X_full[t_idx]
aligned_twitter_y = twitter_y_full[t_idx]

label_match = (aligned_stock_y == aligned_twitter_y)
print(f"   Label match: {np.sum(label_match)}/{len(label_match)} ({100*np.mean(label_match):.1f}%)")

keep_mask = label_match
aligned_stock_X = aligned_stock_X[keep_mask]
aligned_twitter_X = aligned_twitter_X[keep_mask]
aligned_y = aligned_stock_y[keep_mask].astype(np.float32)
aligned_dates = aligned_dates[keep_mask]
aligned_tickers = aligned_tickers[keep_mask]

print(f"\n   After filtering:")
print(f"   Samples: {len(aligned_y)} | Up={int(np.sum(aligned_y))} Down={len(aligned_y) - int(np.sum(aligned_y))}")
print(f"   Date range: {pd.Timestamp(aligned_dates[0]).date()} to {pd.Timestamp(aligned_dates[-1]).date()}")

print("\n3. Chronological train/test split...")

total = len(aligned_y)
train_size = int(total * 0.8)

train_idx = np.arange(0, train_size)
test_idx = np.arange(train_size, total)

print(f"   Train: {len(train_idx)} (80%) | Test: {len(test_idx)} (20%)")

stock_X_tensor = torch.tensor(aligned_stock_X, dtype=torch.float32)
twitter_X_tensor = torch.tensor(aligned_twitter_X, dtype=torch.float32)
y_tensor = torch.tensor(aligned_y, dtype=torch.float32).unsqueeze(1)

stock_X_train = stock_X_tensor[train_idx].to(device)
twitter_X_train = twitter_X_tensor[train_idx].to(device)
y_train = y_tensor[train_idx].to(device)

stock_X_test = stock_X_tensor[test_idx].to(device)
twitter_X_test = twitter_X_tensor[test_idx].to(device)
y_test = y_tensor[test_idx].to(device)

batch_size = 64
train_loader = DataLoader(TensorDataset(stock_X_train, twitter_X_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(stock_X_test, twitter_X_test, y_test), batch_size=batch_size, shuffle=False)
print(f"   Batch size: {batch_size}")

print("\n4. Building models...")

stock_lstm = StockLSTM(input_size=stock_num_features, hidden_size=128, num_layers=2, dropout=0.2).to(device)
twitter_lstm = TwitterLSTM(input_size=twitter_num_features, hidden_size=64, num_layers=2, dropout=0.4).to(device)

fusion_model = HybridFusionModel(dropout_rate=0.2).to(device)

print("   Encoders initialized | Fusion head ready")

print("\n5. Loading pre-trained encoders...")

def load_pretrained_lstm_with_attention(lstm_model, checkpoint_path):
    """Load pretrained LSTM weights."""
    if not os.path.exists(checkpoint_path):
        print(f"   ⚠ WARNING: Missing checkpoint -> {checkpoint_path}")
        return False

    ckpt = torch.load(checkpoint_path, map_location=device)

    new_state = {}
    for k, v in ckpt.items():
        if k.startswith("lstm.lstm."):
            new_state[k.replace("lstm.lstm.", "lstm.")] = v
        elif k.startswith("lstm.attention."):
            new_state[k.replace("lstm.attention.", "attention.")] = v

    if len(new_state) == 0:
        print(f"   ⚠ WARNING: No matching weights found in checkpoint: {checkpoint_path}")
        return False

    lstm_model.load_state_dict(new_state, strict=False)
    print(f"   ✓ Loaded from: {checkpoint_path}")
    return True

stock_checkpoint = 'models/trained/stock_lstm_trained.pth' if os.path.exists('models') else 'trained/stock_lstm_trained.pth'
twitter_checkpoint = 'models/trained/twitter_lstm_trained.pth' if os.path.exists('models') else 'trained/twitter_lstm_trained.pth'

load_pretrained_lstm_with_attention(stock_lstm, stock_checkpoint)
load_pretrained_lstm_with_attention(twitter_lstm, twitter_checkpoint)

print("\n6. Freezing LSTM feature extractors...")

for p in stock_lstm.parameters():
    p.requires_grad = False
for p in twitter_lstm.parameters():
    p.requires_grad = False

stock_lstm.eval()
twitter_lstm.eval()

print("   ✓ StockLSTM frozen")
print("   ✓ TwitterLSTM frozen")

class CompletePipeline(nn.Module):
    """Frozen encoders + trainable fusion head."""
    def __init__(self, stock_lstm, twitter_lstm, fusion_model):
        super().__init__()
        self.stock_lstm = stock_lstm
        self.twitter_lstm = twitter_lstm
        self.fusion_model = fusion_model

    def forward(self, stock_x, twitter_x):
        with torch.no_grad():
            stock_features = self.stock_lstm(stock_x)
            twitter_features = self.twitter_lstm(twitter_x)

        logits = self.fusion_model(stock_features, twitter_features)
        return logits

complete_model = CompletePipeline(stock_lstm, twitter_lstm, fusion_model).to(device)

trainable_params = sum(p.numel() for p in fusion_model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in complete_model.parameters())
print(f"\n   Params: {total_params:,} total | {trainable_params:,} trainable (fusion only)")

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(fusion_model.parameters(), lr=0.001)

print("\n7. Training fusion...")
print("-" * 70)

num_epochs = 50
best_test_acc = 0
best_epoch = 0

for epoch in range(num_epochs):
    fusion_model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_stock, batch_twitter, batch_y in train_loader:
        optimizer.zero_grad()

        logits = complete_model(batch_stock, batch_twitter)
        loss = criterion(logits, batch_y)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        train_correct += (preds == batch_y).sum().item()
        train_total += batch_y.size(0)

    train_loss /= len(train_loader)
    train_acc = 100.0 * train_correct / train_total

    fusion_model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for batch_stock, batch_twitter, batch_y in test_loader:
            logits = complete_model(batch_stock, batch_twitter)
            loss = criterion(logits, batch_y)

            test_loss += loss.item()
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            test_correct += (preds == batch_y).sum().item()
            test_total += batch_y.size(0)

    test_loss /= len(test_loader)
    test_acc = 100.0 * test_correct / test_total
    
    # Save best checkpoint
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_epoch = epoch + 1
        save_path = 'models/trained/hybrid_fusion_trained.pth' if os.path.exists('models') else 'trained/hybrid_fusion_trained.pth'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(fusion_model.state_dict(), save_path)
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:5.1f}% | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:5.1f}%")

print("-" * 70)
print(f"\nBest test accuracy: {best_test_acc:.1f}% (epoch {best_epoch})")

print("\n8. Evaluating best fusion model...")

load_path = 'models/trained/hybrid_fusion_trained.pth' if os.path.exists('models') else 'trained/hybrid_fusion_trained.pth'
fusion_model.load_state_dict(torch.load(load_path, map_location=device))
fusion_model.eval()

with torch.no_grad():
    test_logits = complete_model(stock_X_test, twitter_X_test)
    test_probs = torch.sigmoid(test_logits)
    test_preds = (test_probs > 0.5).float()
    test_acc = 100.0 * (test_preds == y_test).float().mean().item()

print(f"   Test accuracy:  {test_acc:.1f}%")

print("\n9. Sample predictions (test set):")
print("-" * 70)
print(f"{'Date':<12} {'Ticker':<8} {'Prediction':<12} {'Actual':<10} {'Probability':<12}")
print("-" * 70)

test_dates = aligned_dates[test_idx]
test_tickers = aligned_tickers[test_idx]

with torch.no_grad():
    for i in range(min(10, len(test_idx))):
        prob = test_probs[i].item()
        pred_label = "Up" if prob > 0.5 else "Down"
        actual_label = "Up" if y_test[i].item() == 1 else "Down"
        date_str = pd.Timestamp(test_dates[i]).strftime("%Y-%m-%d")
        ticker_str = str(test_tickers[i])

        print(f"{date_str:<12} {ticker_str:<8} {pred_label:<12} {actual_label:<10} {prob:.4f}")

print("\n" + "=" * 70)
print("Hybrid Fusion Training Complete!")
print("=" * 70)
print(f"\nFusion model saved to: models/trained/hybrid_fusion_trained.pth")
print("\nArchitecture Summary:")
print("  • Stock LSTM (frozen): 60-day price seq → 128-dim features")
print("  • Twitter LSTM (frozen): 72-hour sentiment → 64-dim features")
print("  • Fusion head (trained): concat(192) → logits → sigmoid for probability")
