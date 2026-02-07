import torch
import torch.nn as nn


class TwitterLSTM(nn.Module):
    """
    Twitter LSTM encoder for 72-hour sentiment sequences.
    
    Uses attention to extract temporal patterns from hourly aggregated sentiment.
    Output: 64-dim or 128-dim embedding (configurable via hidden_size).
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
        Args:
            x: Sentiment sequences (batch, 72, input_size)
        
        Returns:
            Context vector (batch, hidden_size)
        """
        assert x.dim() == 3 and x.size(1) == 72 and x.size(2) == self.input_size, \
            f"Expected (batch, 72, {self.input_size}), got {x.shape}"
        
        # LSTM + attention
        lstm_out, (h_n, c_n) = self.lstm(x)
        attn_scores = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)
        
        return context_vector
    
    def get_output_dim(self):
        """Returns output dimension."""
        return self.hidden_size