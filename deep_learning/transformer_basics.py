"""Learn positional embeddings and padding masks on synthetic transactions.

Each window ends at its target transaction, so final-transaction classification
does not need a causal mask. This example does not measure generalization.
"""

import torch
from torch import nn


class FraudTransformer(nn.Module):
    def __init__(self, input_size=2, embedding_dim=16, num_heads=2, max_length=3):
        super().__init__()
        self.projection = nn.Linear(input_size, embedding_dim)
        self.positions = nn.Embedding(max_length, embedding_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=32,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=1, enable_nested_tensor=False
        )
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, x, lengths, padding_mask):
        # Right padding: False marks real events; True marks padded positions.
        positions = torch.arange(x.size(1), device=x.device)
        embedded = self.projection(x) + self.positions(positions)
        encoded = self.encoder(embedded, src_key_padding_mask=padding_mask)
        batch_indices = torch.arange(x.size(0), device=x.device)
        last_indices = lengths.to(x.device) - 1
        final_transaction = encoded[batch_indices, last_indices]
        return self.classifier(final_transaction)


def main():
    torch.manual_seed(42)
    # Features: normalized amount and time gap. Labels refer to the final event.
    sequences = [
        [[0.1, 0.5], [0.2, 0.4], [0.9, 0.01]],
        [[0.2, 0.6], [0.1, 0.7]],
        [[0.3, 0.5], [0.9, 0.02], [0.95, 0.01]],
        [[0.1, 0.8], [0.2, 0.9]],
    ]
    labels = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    lengths = torch.tensor([len(sequence) for sequence in sequences])
    max_length = lengths.max().item()
    X = torch.zeros(len(sequences), max_length, 2)
    padding_mask = torch.ones(len(sequences), max_length, dtype=torch.bool)
    for i, sequence in enumerate(sequences):
        X[i, :len(sequence)] = torch.tensor(sequence, dtype=torch.float32)
        padding_mask[i, :len(sequence)] = False

    print("Input shape:", tuple(X.shape))
    print("Sequence lengths:", lengths.tolist())
    print("Padding mask (True = padding):")
    print(padding_mask)
    model = FraudTransformer(max_length=max_length)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(200):
        model.train()
        logits = model(X, lengths, padding_mask)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch == 0 or (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(X, lengths, padding_mask))
    print("\nPredicted fraud probabilities:")
    print(probabilities.squeeze(-1))
    print("\nOutput shape:", tuple(probabilities.shape))
    print("\nSynthetic training examples only: this is not a generalization test.")


if __name__ == "__main__":
    main()
