"""
model.py – Deep learning architectures for resume text classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMClassifier(nn.Module):
    """
    Bidirectional 2-layer LSTM text classifier.
    Architecture: Embedding → BiLSTM(×2) → Dropout → Linear

    Chosen because:
    - Bidirectional captures left/right context in resume sentences
    - 2 layers learn hierarchical text representations
    - Runs in < 5 min on CPU with the dataset sizes used here
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # ×2 for bidirectional

    def forward(self, x):
        emb = self.dropout(self.embedding(x))           # (B, L, E)
        _, (hidden, _) = self.lstm(emb)                 # hidden: (2*layers, B, H)
        # Last forward (hidden[-2]) + last backward (hidden[-1])
        out = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (B, H*2)
        return self.fc(self.dropout(out))               # (B, C)


class GRUClassifier(nn.Module):
    """
    Bidirectional GRU variant – ~30% faster than LSTM on CPU.
    Used as a secondary experiment to compare performance/speed trade-off.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        _, hidden = self.gru(emb)
        out = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(out))


def count_parameters(model: nn.Module) -> int:
    """Return number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
