"""Learn GRU shapes and right padding with four synthetic sequences.

This fits the training examples only; it does not measure fraud generalization.
No real transaction data is loaded.
"""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class FraudGRU(nn.Module):
    def __init__(self, input_size=2, hidden_size=8):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths):
        # x: (batch, padded sequence length, features).
        # Lengths describe valid prefixes; padded suffixes are not observations.
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)
        # h_n: (recurrent layers, batch, hidden size), here (1, 4, 8).
        final_hidden = h_n[-1]
        logits = self.classifier(final_hidden)
        return logits


def main():
    torch.manual_seed(42)

    # Two invented, scaled features per transaction: amount and activity level.
    # Labels describe whole sequences, not individual time steps.
    X = torch.tensor([
        [[0.1, 0.2], [0.2, 0.1], [0.1, 0.1]],
        [[0.8, 0.7], [0.9, 0.9], [0.0, 0.0]],
        [[0.2, 0.1], [0.1, 0.3], [0.3, 0.2]],
        [[0.7, 0.8], [1.0, 0.9], [0.0, 0.0]],
    ], dtype=torch.float32)
    lengths = torch.tensor([3, 2, 3, 2], dtype=torch.int64)
    labels = torch.tensor([[0.0], [1.0], [0.0], [1.0]])

    print("Input shape:", tuple(X.shape))
    print("Sequence lengths:", lengths.tolist())
    model = FraudGRU(input_size=2, hidden_size=8)
    # BCEWithLogitsLoss applies the sigmoid internally in a numerically stable way.
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(200):
        model.train()
        logits = model(X, lengths)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch == 0 or (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(X, lengths))
    print("\nPredicted fraud probabilities:")
    print(probabilities.squeeze(-1))
    print("\nOutput shape:", tuple(probabilities.shape))
    print("\nSynthetic training examples only: this is not a generalization test.")


if __name__ == "__main__":
    main()
