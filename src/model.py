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


class AttentionBiLSTM(nn.Module):
    """
    BiLSTM with additive self-attention.

    Why: standard BiLSTM uses only the final hidden state, so the last words
    of the resume have disproportionate influence. Attention computes a
    learned weighted average over ALL token positions — the model learns
    to focus on the most informative words (e.g. "Python", "surgeon", "teacher").

    Architecture:
        Embedding → Dropout → BiLSTM(2 layers) → Attention → Dropout → Linear
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
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb     = self.dropout(self.embedding(x))   # (B, L, E)
        out, _  = self.lstm(emb)                    # (B, L, H*2)
        scores  = self.attention(out)               # (B, L, 1)
        weights = torch.softmax(scores, dim=1)      # (B, L, 1)  — sum over L = 1
        context = (out * weights).sum(dim=1)        # (B, H*2)   — weighted avg
        return self.fc(self.dropout(context))       # (B, C)


class TextCNN(nn.Module):
    """
    Convolutional Neural Network for text classification (Kim 2014).

    Uses parallel Conv1d filters of different sizes to detect local n-gram
    patterns at multiple scales, then max-over-time pooling to pick the
    most relevant signal per filter.

    Architecture:
        Embedding → [Conv(2), Conv(3), Conv(4), Conv(5)] → MaxPool → Concat → Dropout → Linear
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        num_filters: int = 128,
        filter_sizes: tuple = (2, 3, 4, 5),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        emb = self.embedding(x).transpose(1, 2)        # (B, E, L) — Conv1d expects (B, C, L)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(emb))                      # (B, F, L-fs+1)
            p = c.max(dim=2)[0]                        # (B, F) — max-over-time pooling
            pooled.append(p)
        out = torch.cat(pooled, dim=1)                 # (B, F * n_filters)
        return self.fc(self.dropout(out))              # (B, C)


def count_parameters(model: nn.Module) -> int:
    """Return number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
