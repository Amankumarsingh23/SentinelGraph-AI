"""Session 14: two-layer GCN on five synthetic nodes.

This is transductive learning: every node's features/edges participate in message
passing, but only train_mask labels contribute to loss. It is not a temporal
fraud experiment and loads no IEEE-CIS data.
"""

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


def make_toy_graph():
    x = torch.tensor([
        [0.1, 0.2],
        [0.2, 0.3],
        [0.9, 0.8],
        [0.8, 0.9],
        [0.3, 0.2],
    ], dtype=torch.float32)
    # Both directions represent the undirected chain 0--1--2--3--4.
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4],
        [1, 0, 2, 1, 3, 2, 4, 3],
    ], dtype=torch.long)
    y = torch.tensor([0, 0, 1, 1, 0], dtype=torch.float32)
    train_mask = torch.tensor([True, True, True, False, False])
    return Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask)


class FraudGCN(nn.Module):
    def __init__(self, input_size=2, hidden_size=8):
        super().__init__()
        # GCNConv adds self-loops and applies symmetric degree normalization.
        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, 1)

    def encode(self, x, edge_index):
        hidden = torch.relu(self.conv1(x, edge_index))
        return self.conv2(hidden, edge_index)

    def forward(self, x, edge_index):
        embeddings = self.encode(x, edge_index)
        return self.classifier(embeddings).squeeze(-1)


def main():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    graph = make_toy_graph()
    graph.validate(raise_on_error=True)
    print("Node feature shape:", tuple(graph.x.shape))
    print("Edge-index shape:", tuple(graph.edge_index.shape))
    print("Label shape:", tuple(graph.y.shape))
    print("Training nodes:", graph.train_mask.nonzero().flatten().tolist())
    model = FraudGCN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        logits = model(graph.x, graph.edge_index)
        # Mask BOTH logits and labels. Unmasked labels never affect gradients.
        loss = loss_fn(logits[graph.train_mask], graph.y[graph.train_mask])
        loss.backward()
        optimizer.step()
        if epoch == 0 or (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}, Training loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        embeddings = model.encode(graph.x, graph.edge_index)
        probabilities = torch.sigmoid(model(graph.x, graph.edge_index))
    print("\nNode embedding shape:", tuple(embeddings.shape))
    print("Predicted probabilities:", probabilities)
    print("Final prediction shape:", tuple(probabilities.shape))
    print("\nSynthetic transductive demonstration only; these are not validated fraud probabilities.")
    print("train_mask controls loss labels; it does not hide nodes or edges from message passing.")


if __name__ == "__main__":
    main()
