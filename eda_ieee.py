
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# 1. Configure paths
DATA_PATH = Path("data/raw/train_transaction.csv")
OUTPUT_DIR = Path("outputs/eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Read a manageable subset of columns
columns = [
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card1",
    "isFraud",
]

df = pd.read_csv(DATA_PATH, usecols=columns)

# 3. Basic dataset inspection
print("Dataset shape:", df.shape)
print("\nColumn types:")
print(df.dtypes)

print("\nFirst five transactions:")
print(df.head())

# 4. Fraud distribution
print("\nClass counts:")
print(df["isFraud"].value_counts())

fraud_rate = df["isFraud"].mean()
print(f"\nFraud rate: {fraud_rate:.2%}")

# 5. Missingness percentages
missing_pct = df.isna().mean() * 100
print("\nMissingness percentage:")
print(missing_pct.sort_values(ascending=False))

# 6. Amount statistics by fraud label
print("\nTransaction amount statistics:")
print(
    df.groupby("isFraud")["TransactionAmt"]
      .agg(["count", "mean", "median", "max"])
)

# 7. Observed fraud rate by product code
product_summary = (
    df.groupby("ProductCD", dropna=False)
      .agg(
          transactions=("isFraud", "size"),
          fraud_rate=("isFraud", "mean")
      )
)

print("\nFraud rate by product category:")
print(product_summary)

# 8. Save a summary
product_summary.to_csv(
    OUTPUT_DIR / "product_summary.csv"
)

# 9. Visualize class distribution
df["isFraud"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Fraud vs Legitimate Transactions")
plt.xlabel("Fraud Label (0 = Legitimate, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "class_distribution.png",
    dpi=150
)
plt.close()

# 10. Analyze fraud over relative time
df = df.sort_values("TransactionDT").copy()

# Bin relative time into approximate days
df["relative_day"] = df["TransactionDT"] // 86400

daily_summary = (
    df.groupby("relative_day")
      .agg(
          transaction_count=("isFraud", "size"),
          fraud_rate=("isFraud", "mean")
      )
)

daily_summary.to_csv(
    OUTPUT_DIR / "daily_summary.csv"
)

daily_summary["fraud_rate"].plot()
plt.title("Observed Fraud Rate Over Relative Time")
plt.xlabel("Days Since Dataset Reference Point")
plt.ylabel("Fraud Rate")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "fraud_over_time.png",
    dpi=150
)
plt.close()

print("\nEDA completed successfully!")
print(f"Results saved in: {OUTPUT_DIR}")
