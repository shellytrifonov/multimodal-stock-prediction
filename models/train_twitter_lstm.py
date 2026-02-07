import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from models.twitter_lstm import TwitterLSTM

print("=" * 70)
print("Twitter LSTM Training")
print("=" * 70)

print("\n1. Loading training data...")
import os

if os.path.exists('twitter_lstm_training_data.npz'):
    data_path = 'twitter_lstm_training_data.npz'
else:
    data_path = '../twitter_lstm_training_data.npz'

if not os.path.exists(data_path):
    print(f"   ERROR: Data not found at {data_path}")
    print("   Run build_twitter_training_data.py first")
    exit(1)

data = np.load(data_path)

X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']
num_features = int(data['num_features'])

print(f"   Train: {len(X_train)} | Test: {len(X_test)}")
print(f"   Shape: {X_train.shape} (72 hours × {num_features} features)")
print(f"   Train labels: {np.sum(y_train)} Up / {len(y_train) - np.sum(y_train)} Down")
print(f"   Test labels: {np.sum(y_test)} Up / {len(y_test) - np.sum(y_test)} Down")

X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).unsqueeze(1)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).unsqueeze(1)

print("   Chronological split")

print("\n2. Creating data loaders...")
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=8, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=8, shuffle=False)
print(f"   Batch size: 8 (small to reduce overfitting)")

print("\n3. Building model...")
# Reduced hidden_size to 64 and increased dropout to 0.4 to prevent overfitting
model = TwitterLSTM(
    input_size=num_features,
    hidden_size=64,
    num_layers=2,
    dropout=0.4
)
print(f"   LSTM: {num_features} -> 64 (hidden_size reduced to prevent overfitting)")
print(f"   Dropout: 0.4 (stronger regularization)")

classifier = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(16, 1),
    nn.Sigmoid()
)

class FullModel(nn.Module):
    def __init__(self, lstm, classifier):
        super().__init__()
        self.lstm = lstm
        self.classifier = classifier
    
    def forward(self, x):
        features = self.lstm(x)
        output = self.classifier(features)
        return output

full_model = FullModel(model, classifier)

print(f"   Params: {sum(p.numel() for p in full_model.parameters()):,}")

print("\n4. Training setup...")
criterion = nn.BCELoss()
optimizer = optim.Adam(full_model.parameters(), lr=0.001, weight_decay=1e-4)  # L2 regularization
print(f"   Loss: BCELoss | Optimizer: Adam (lr=0.001, L2=1e-4) | Epochs: 100")

print("\n5. Training...")
print("-" * 70)

num_epochs = 100
best_test_acc = 0
best_epoch = 0

for epoch in range(num_epochs):
    full_model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = full_model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        predictions = (outputs > 0.5).float()
        train_correct += (predictions == batch_y).sum().item()
        train_total += batch_y.size(0)
    
    train_loss /= len(train_loader)
    train_acc = 100 * train_correct / train_total
    
    full_model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = full_model(batch_X)
            loss = criterion(outputs, batch_y)
            
            test_loss += loss.item()
            predictions = (outputs > 0.5).float()
            test_correct += (predictions == batch_y).sum().item()
            test_total += batch_y.size(0)
    
    test_loss /= len(test_loader)
    test_acc = 100 * test_correct / test_total
    
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_epoch = epoch + 1
        save_path = 'models/trained/twitter_lstm_trained.pth' if os.path.exists('models') else 'trained/twitter_lstm_trained.pth'
        torch.save(full_model.state_dict(), save_path)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}/{num_epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:5.1f}% | Test Loss: {test_loss:.4f} Acc: {test_acc:5.1f}%")

print("-" * 70)
print(f"\nBest test accuracy: {best_test_acc:.1f}% (epoch {best_epoch})")

print("\n6. Final evaluation...")
load_path = 'models/trained/twitter_lstm_trained.pth' if os.path.exists('models') else 'trained/twitter_lstm_trained.pth'
full_model.load_state_dict(torch.load(load_path))
full_model.eval()

with torch.no_grad():
    train_preds = full_model(X_train)
    test_preds = full_model(X_test)
    
    train_pred_labels = (train_preds > 0.5).float()
    test_pred_labels = (test_preds > 0.5).float()
    
    train_acc = 100 * (train_pred_labels == y_train).float().mean().item()
    test_acc = 100 * (test_pred_labels == y_test).float().mean().item()

print(f"Train accuracy: {train_acc:.1f}% | Test accuracy: {test_acc:.1f}% | Gap: {train_acc - test_acc:.1f}%")

print("\n7. Sample predictions:")
print("-" * 70)
print(f"{'Predicted':<12} {'Actual':<10} {'Prob':<10}")
print("-" * 70)

with torch.no_grad():
    for i in range(min(5, len(X_test))):
        pred_prob = test_preds[i].item()
        pred_label = "Up" if pred_prob > 0.5 else "Down"
        actual_label = "Up" if y_test[i].item() == 1 else "Down"
        
        print(f"{pred_label:<12} {actual_label:<10} {pred_prob:.4f}")

print("\n" + "=" * 70)
print("Training complete!")
print("=" * 70)
print(f"Saved: models/trained/twitter_lstm_trained.pth")
print(f"Test accuracy: {test_acc:.1f}%")