import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from models.stock_lstm import StockLSTM

print("=" * 70)
print("Stock LSTM Training")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

print("\n1. Loading training data...")
import os

if os.path.exists('stock_lstm_training_data.npz'):
    data_path = 'stock_lstm_training_data.npz'
else:
    data_path = '../stock_lstm_training_data.npz'

if not os.path.exists(data_path):
    print(f"   ERROR: Data not found at {data_path}")
    print("   Run build_stock_training_data.py first")
    exit(1)

data = np.load(data_path)

X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']

num_features = X_train.shape[2]
print(f"   Train: {len(X_train)} | Test: {len(X_test)}")
print(f"   Shape: {X_train.shape} ({X_train.shape[1]} days × {num_features} features)")
if num_features == 25:
    print(f"   Features: OHLCV + log_volume + 19 technical indicators")
print(f"   Train labels: {np.sum(y_train)} Up / {len(y_train) - np.sum(y_train)} Down")
print(f"   Test labels: {np.sum(y_test)} Up / {len(y_test) - np.sum(y_test)} Down")

if X_train.shape[1] != 60:
    print(f"   ⚠ WARNING: Expected 60-day sequences, got {X_train.shape[1]}")
if num_features == 6:
    print(f"   ⚠ WARNING: Using old 6-feature data, rebuild training data")
elif num_features != 25:
    print(f"   ⚠ WARNING: Expected 25 features, got {num_features}")

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

print(f"   Chronological split | Moved to {device}")

print("\n2. Creating data loaders...")
batch_size = 64
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
print(f"   Batch size: {batch_size}")

print("\n3. Building model...")

feature_extractor = StockLSTM(
    input_size=num_features,
    hidden_size=128,
    num_layers=2,
    dropout=0.2
).to(device)

classifier = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(32, 1)
).to(device)

class FullModel(nn.Module):
    def __init__(self, lstm, classifier):
        super().__init__()
        self.lstm = lstm
        self.classifier = classifier

    def forward(self, x):
        features = self.lstm(x)
        logits = self.classifier(features)
        return logits

full_model = FullModel(feature_extractor, classifier).to(device)

print(f"   LSTM: {num_features} -> {feature_extractor.get_output_dim()} | Params: {sum(p.numel() for p in full_model.parameters()):,}")

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(full_model.parameters(), lr=0.001)

print("\n4. Training...")
print("-" * 70)

num_epochs = 100
best_test_acc = 0
best_epoch = 0

for epoch in range(num_epochs):
    full_model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()

        logits = full_model(batch_X)
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

    full_model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            logits = full_model(batch_X)
            loss = criterion(logits, batch_y)

            test_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            test_correct += (preds == batch_y).sum().item()
            test_total += batch_y.size(0)

    test_loss /= len(test_loader)
    test_acc = 100.0 * test_correct / test_total
    
    # Save best model
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_epoch = epoch + 1
        save_path = 'models/trained/stock_lstm_trained.pth' if os.path.exists('models') else 'trained/stock_lstm_trained.pth'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(full_model.state_dict(), save_path)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:5.1f}% | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:5.1f}%")

print("-" * 70)
print(f"\nBest: {best_test_acc:.1f}% at epoch {best_epoch}")

print("\n5. Final evaluation...")
load_path = 'models/trained/stock_lstm_trained.pth' if os.path.exists('models') else 'trained/stock_lstm_trained.pth'
full_model.load_state_dict(torch.load(load_path, map_location=device))
full_model.eval()

with torch.no_grad():
    train_logits = full_model(X_train)
    test_logits = full_model(X_test)

    train_probs = torch.sigmoid(train_logits)
    test_probs = torch.sigmoid(test_logits)

    train_preds = (train_probs > 0.5).float()
    test_preds = (test_probs > 0.5).float()

    train_acc = 100.0 * (train_preds == y_train).float().mean().item()
    test_acc = 100.0 * (test_preds == y_test).float().mean().item()

print(f"   Train: {train_acc:.1f}% | Test: {test_acc:.1f}%")

print("\n6. Sample predictions:")
print("-" * 70)
print(f"{'Predicted':<12} {'Actual':<10} {'Prob':<10}")
print("-" * 70)

with torch.no_grad():
    for i in range(min(10, len(X_test))):
        prob = test_probs[i].item()
        pred_label = "Up" if prob > 0.5 else "Down"
        actual_label = "Up" if y_test[i].item() == 1 else "Down"
        print(f"{pred_label:<12} {actual_label:<10} {prob:.4f}")

print("\n" + "=" * 70)
print("Training complete!")
print("=" * 70)
print(f"\nSaved: models/trained/stock_lstm_trained.pth")
print("\nNote: Classifier head is temporary for validation.")
print("Production uses only LSTM encoder (128-dim) -> HybridFusionModel.")
