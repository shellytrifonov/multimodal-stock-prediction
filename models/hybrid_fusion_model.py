import torch
import torch.nn as nn


class HybridFusionModel(nn.Module):
    """
    Fusion model combining stock (128-dim) and Twitter (64-dim) features.
    
    Architecture:
        Concat: 192 -> Fusion: 128 -> 64 -> Prediction: 32 -> 1 (logits)
    
    Args:
        dropout_rate: Dropout for regularization (default: 0.2)
    
    Input:
        - stock_features: (batch, 128)
        - twitter_features: (batch, 64)
    
    Output: (batch, 1) logits for BCEWithLogitsLoss
    """
    
    def __init__(self, dropout_rate=0.2):
        super(HybridFusionModel, self).__init__()
        
        self.stock_feature_size = 128
        self.twitter_feature_size = 64
        self.concat_size = 192
        
        # Fusion layers: 192 -> 128 -> 64
        self.fusion_layer = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )
        
        # Prediction head: 64 -> 32 -> 1
        self.prediction_layer = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, stock_features=None, twitter_features=None):
        """
        Args:
            stock_features: (batch, 128) or None
            twitter_features: (batch, 64)
        
        Returns:
            logits (batch, 1)
        """
        if twitter_features is None:
            raise ValueError("twitter_features required")
        
        batch_size = twitter_features.size(0)
        device = twitter_features.device
        
        # Zero-fill stock features if missing
        if stock_features is None:
            stock_features = torch.zeros(batch_size, 128, device=device)
        
        # Concatenate features: (batch, 192)
        combined = torch.cat([stock_features, twitter_features], dim=1)
        
        # Fusion: (batch, 192) -> (batch, 64)
        fused = self.fusion_layer(combined)
        
        # Prediction: (batch, 64) -> (batch, 1)
        logits = self.prediction_layer(fused)
        
        return logits
    
    def predict(self, stock_features=None, twitter_features=None, threshold=0.5):
        """Generate binary predictions with configurable threshold.
        
        Returns:
            (predictions, probabilities) tuple
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(stock_features, twitter_features)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= threshold).float()
        
        return predictions, probabilities


def test_model():
    """Quick test for fusion model."""
    print("=" * 70)
    print("Testing HybridFusionModel")
    print("=" * 70)
    
    model = HybridFusionModel(dropout_rate=0.2)
    print(f"\nModel initialized | Params: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n1. Forward pass with both features...")
    batch_size = 4
    stock_feat = torch.randn(batch_size, 128)
    twitter_feat = torch.randn(batch_size, 64)
    
    output = model(stock_feat, twitter_feat)
    print(f"   Stock {stock_feat.shape} + Twitter {twitter_feat.shape} -> {output.shape}")
    assert output.shape == (batch_size, 1)
    print("   ✓ Passed")
    
    print("\n2. Testing with stock_features=None...")
    output = model(stock_features=None, twitter_features=twitter_feat)
    assert output.shape == (batch_size, 1)
    print("   ✓ Passed")
    
    print("\n3. Predict method test...")
    predictions, probabilities = model.predict(stock_feat, twitter_feat)
    print(f"   Preds: {predictions.squeeze().tolist()}")
    print("   ✓ Passed")
    
    print("\n4. Twitter-only test...")
    output = model(twitter_features=twitter_feat)
    assert output.shape == (batch_size, 1)
    print("   ✓ Passed")
    
    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
    print("\nArchitecture:")
    print(model)


if __name__ == "__main__":
    test_model()
