"""Small models reusing the Session 9 GRU and Session 10 Transformer."""

import torch
from torch import nn
from deep_learning.gru_basics import FraudGRU
from deep_learning.transformer_basics import FraudTransformer


def make_model(kind, input_size, max_length, hidden_size=32):
    if kind == "mlp":
        return nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(),
                             nn.Linear(hidden_size, 1))
    if kind == "gru":
        return FraudGRU(input_size=input_size, hidden_size=hidden_size)
    if kind == "transformer":
        return FraudTransformer(input_size=input_size, embedding_dim=hidden_size,
                                num_heads=4, max_length=max_length)
    raise ValueError(f"Unknown model: {kind}")


def forward(model, kind, x, lengths):
    if kind == "mlp":
        return model(x[:, 0])
    if kind == "gru":
        return model(x, lengths)
    padding = torch.arange(x.size(1), device=x.device)[None, :] >= lengths[:, None]
    return model(x, lengths, padding)
