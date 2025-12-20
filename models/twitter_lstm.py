import torch
import torch.nn as nn


class TwitterLSTM(nn.Module):
    """
    Attention-based LSTM for temporal sentiment sequence encoding.
    Processes 72-hour sentiment windows to extract temporal patterns.
    """
    
    def __init__(self, input_size=5, hidden_size=128, num_layers=2, dropout=0.2):
        super(TwitterLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        """
        Forward pass with attention mechanism.
        
        Args:
            x: Input tensor (batch_size, 72, num_features)
        
        Returns:
            Context vector (batch_size, hidden_size)
        """
        assert x.dim() == 3, f"Expected 3D input, got {x.dim()}D"
        assert x.size(1) == 72, f"Expected 72 timesteps, got {x.size(1)}"
        assert x.size(2) == self.input_size, f"Expected {self.input_size} features, got {x.size(2)}"
        
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        attn_scores = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)
        
        return context_vector
    
    def get_output_dim(self):
        """Returns output dimensionality."""
        return self.hidden_size