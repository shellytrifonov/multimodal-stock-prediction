import torch
import torch.nn as nn


class StockLSTM(nn.Module):
    """
    Stock LSTM encoder for 60-day price sequences.
    
    Extracts temporal features from historical stock data using 2-layer LSTM
    with attention mechanism. Outputs 128-dim embedding for fusion model.
    
    Args:
        input_size: Features per day (6 for OHLCV, 25 with technical indicators)
        hidden_size: LSTM hidden dimension (default: 128)
        num_layers: LSTM depth (default: 2)
        dropout: Regularization rate (default: 0.2)
    
    Input: (batch, 60, input_size)
    Output: (batch, 128)
    """
    
    def __init__(self, input_size=6, hidden_size=128, num_layers=2, dropout=0.2):
        super(StockLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 2-layer LSTM with dropout between layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Attention layer to weight timesteps by importance
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        """
        Args:
            x: Stock sequences (batch, 60, input_size)
        
        Returns:
            Context vector (batch, 128)
        """
        # Input validation
        assert x.dim() == 3, f"Expected 3D input, got {x.dim()}D"
        assert x.size(1) == 60, f"Expected 60 days, got {x.size(1)}"
        assert x.size(2) == self.input_size, f"Expected {self.input_size} features, got {x.size(2)}"
        
        # LSTM forward pass: (batch, 60, input_size) -> (batch, 60, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Compute attention weights over timesteps
        attn_scores = self.attention(lstm_out)  # (batch, 60, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        
        # Weighted sum of LSTM outputs
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden_size)
        
        return context_vector
    
    def get_output_dim(self):
        """Returns output dimension (128)."""
        return self.hidden_size
    
    def get_attention_weights(self, x):
        """Extract attention weights for visualization.
        
        Useful for interpreting which days matter most for predictions.
        """
        self.eval()
        with torch.no_grad():
            lstm_out, _ = self.lstm(x)
            attn_scores = self.attention(lstm_out)
            attn_weights = torch.softmax(attn_scores, dim=1)
        
        return attn_weights


def test_model():
    """Quick test to verify model works correctly."""
    print("=" * 70)
    print("Testing StockLSTM Model")
    print("=" * 70)
    
    model = StockLSTM(input_size=6, hidden_size=128, num_layers=2, dropout=0.2)
    print(f"\nModel initialized | Params: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n1. Forward pass test...")
    batch_size = 8
    x = torch.rand(batch_size, 60, 6)  # Random normalized data
    
    output = model(x)
    print(f"   Input: {x.shape} -> Output: {output.shape}")
    assert output.shape == (batch_size, 128), f"Shape mismatch: got {output.shape}"
    print("   ✓ Passed")
    
    print("\n2. Output dimension check...")
    assert model.get_output_dim() == 128
    print("   ✓ Passed")
    
    print("\n3. Attention weights test...")
    attn_weights = model.get_attention_weights(x)
    print(f"   Shape: {attn_weights.shape} | Sum: {attn_weights[0].sum():.4f}")
    assert attn_weights.shape == (batch_size, 60, 1)
    print("   ✓ Passed")
    
    print("\n4. Variable batch size test...")
    for bs in [1, 4, 16, 32]:
        output_test = model(torch.rand(bs, 60, 6))
        assert output_test.shape == (bs, 128)
    print("   ✓ Passed")
    
    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
    print("\nModel architecture:")
    print(model)
    print("\nUsage: stock_features = stock_lstm(x)  # (batch, 60, 6) -> (batch, 128)")


if __name__ == "__main__":
    test_model()
