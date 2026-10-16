
import torch
from torch import nn

torch.manual_seed(42)

# 1. Five example transactions, three features each
# These are synthetic learning examples.
X = torch.tensor([
    [0.2, 0.1, 0.0],
    [0.9, 0.8, 1.0],
    [0.3, 0.2, 0.0],
    [0.8, 0.9, 1.0],
    [0.4, 0.1, 0.0],
], dtype=torch.float32)

y = torch.tensor([
    [0.0], [1.0], [0.0], [1.0], [0.0]
])

# 2. Define our first neural network
model = nn.Sequential(
    nn.Linear(3, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)

# 3. Loss function for binary classification
loss_fn = nn.BCEWithLogitsLoss()

# 4. Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

# 5. Training loop
for epoch in range(200):

    model.train()

    logits = model(X)
    loss = loss_fn(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(
            f"Epoch {epoch+1}: "
            f"Loss = {loss.item():.4f}"
        )

# 6. Inspect predicted probabilities
model.eval()

with torch.no_grad():
    logits = model(X)
    probabilities = torch.sigmoid(logits)

print("\nPredicted probabilities:")
print(probabilities)

print("\nInput shape:", X.shape)
print("Output shape:", probabilities.shape)
