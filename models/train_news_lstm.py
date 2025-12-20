import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from models.news_lstm import NewsLSTM

print("=" * 70)
print("Training News LSTM")
print("=" * 70)

print("\n1. Loading data...")
import os

if os.path.exists('news_lstm_training_data.npz'):
    data_path = 'news_lstm_training_data.npz'
else:
    data_path = '../news_lstm_training_data.npz'

if not os.path.exists(data_path):
    print(f"   ERROR: Training data not found at {data_path}")
    print("   Run: python pipelines/news/build_news_training_data.py first")
    exit(1)

data = np.load(data_path)
X = data['X']
y = data['y']
dates = data['dates']

num_samples, seq_len, num_features = X.shape

print(f"   Total samples: {num_samples}")
print(f"   Input shape: {X.shape}")
print(f"   Features per hour: {num_features}")
print(f"   Labels: {np.sum(y)} Up, {len(y) - np.sum(y)} Down")

if num_features != 8:
    print(f"   WARNING: Expected 8 features but got {num_features}!")

X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y).unsqueeze(1)

print("\n2. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_tensor, y_tensor, test_size=0.2, random_state=42, stratify=y
)

print(f"   Train: {len(X_train)} samples")
print(f"   Test:  {len(X_test)} samples")

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

print("\n3. Initializing model...")
model = NewsLSTM(
    input_size=num_features,
    hidden_size=128,
    num_layers=2,
    dropout=0.2
)
print(f"   Input features: {num_features}")

classifier = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(32, 1),
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

print(f"   Parameters: {sum(p.numel() for p in full_model.parameters()):,}")

criterion = nn.BCELoss()
optimizer = optim.Adam(full_model.parameters(), lr=0.001)

print("\n4. Training...")
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
        save_path = 'models/trained/news_lstm_trained.pth' if os.path.exists('models') else 'trained/news_lstm_trained.pth'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(full_model.state_dict(), save_path)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:5.1f}% | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:5.1f}%")

print("-" * 70)
print(f"\nBest test accuracy: {best_test_acc:.1f}% (epoch {best_epoch})")

print("\n5. Evaluating best model...")
load_path = 'models/trained/news_lstm_trained.pth' if os.path.exists('models') else 'trained/news_lstm_trained.pth'
full_model.load_state_dict(torch.load(load_path))
full_model.eval()

with torch.no_grad():
    train_preds = full_model(X_train)
    test_preds = full_model(X_test)
    
    train_pred_labels = (train_preds > 0.5).float()
    test_pred_labels = (test_preds > 0.5).float()
    
    train_acc = 100 * (train_pred_labels == y_train).float().mean().item()
    test_acc = 100 * (test_pred_labels == y_test).float().mean().item()

print(f"   Train accuracy: {train_acc:.1f}%")
print(f"   Test accuracy:  {test_acc:.1f}%")

print("\n6. Sample predictions (test set):")
print("-" * 70)
print(f"{'Prediction':<12} {'Actual':<10} {'Probability':<12}")
print("-" * 70)

with torch.no_grad():
    for i in range(min(5, len(X_test))):
        pred_prob = test_preds[i].item()
        pred_label = "Up" if pred_prob > 0.5 else "Down"
        actual_label = "Up" if y_test[i].item() == 1 else "Down"
        
        print(f"{pred_label:<12} {actual_label:<10} {pred_prob:.4f}")

print("\n" + "=" * 70)
print("Training Complete!")
print("=" * 70)
print(f"\nModel saved to: models/trained/news_lstm_trained.pth")