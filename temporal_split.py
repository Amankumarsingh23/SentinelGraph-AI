
from pathlib import Path
import numpy as np
import pandas as pd

# 1. Define file locations
DATA_PATH = Path("data/raw/train_transaction.csv")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load the transaction data
df = pd.read_csv(DATA_PATH)

# 3. Sort chronologically
df = df.sort_values(
    ["TransactionDT", "TransactionID"],
    kind="stable"
).reset_index(drop=True)

# 4. Calculate initial split positions
n = len(df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

# Keep identical timestamps in the same split
def move_past_ties(data, boundary):
    if boundary <= 0 or boundary >= len(data):
        return boundary

    time_value = data.iloc[boundary - 1]["TransactionDT"]

    while (
        boundary < len(data)
        and data.iloc[boundary]["TransactionDT"] == time_value
    ):
        boundary += 1

    return boundary

train_end = move_past_ties(df, train_end)
val_end = move_past_ties(df, val_end)

# 5. Split into three chronological datasets
train = df.iloc[:train_end].copy()
val = df.iloc[train_end:val_end].copy()
test = df.iloc[val_end:].copy()

# 6. Inspect the resulting datasets
for name, data in [
    ("Train", train),
    ("Validation", val),
    ("Test", test),
]:
    print(f"\n{name}")
    print("Rows:", len(data))
    print("Fraud rate:", round(data["isFraud"].mean() * 100, 3), "%")
    print("Start:", data["TransactionDT"].min())
    print("End:", data["TransactionDT"].max())

# 7. Check for transaction overlap
train_ids = set(train["TransactionID"])
val_ids = set(val["TransactionID"])
test_ids = set(test["TransactionID"])

assert train_ids.isdisjoint(val_ids)
assert train_ids.isdisjoint(test_ids)
assert val_ids.isdisjoint(test_ids)

# 8. Check chronological ordering
assert train["TransactionDT"].max() < val["TransactionDT"].min()
assert val["TransactionDT"].max() < test["TransactionDT"].min()

# 9. Save the splits
train.to_csv(OUTPUT_DIR / "train.csv", index=False)
val.to_csv(OUTPUT_DIR / "validation.csv", index=False)
test.to_csv(OUTPUT_DIR / "test.csv", index=False)

print("\nTemporal split completed successfully!")
