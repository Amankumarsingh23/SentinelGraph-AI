
import numpy as np
import pandas as pd

# 1. Create a small transaction dataset
data = {
    "transaction_id": ["T001", "T002", "T003", "T004", "T005"],
    "amount": [500, 25000, 1200, 40000, 800],
    "device": ["Mobile", "Laptop", "Mobile", None, "Laptop"],
    "is_fraud": [0, 1, 0, 1, 0],
}

df = pd.DataFrame(data)

# 2. Inspect the dataset
print("Dataset shape:", df.shape)
print("\nFirst rows:")
print(df.head())

# 3. Check missing values
print("\nMissing values:")
print(df.isna().sum())

# 4. Calculate fraud rate
fraud_rate = df["is_fraud"].mean()
print("\nFraud rate:", fraud_rate)

# 5. Explore high-value transactions
high_value = df[df["amount"] > 10000]
print("\nTransactions above 10000:")
print(high_value)

# 6. Engineer a missingness feature
df["device_missing"] = df["device"].isna().astype(int)

# 7. Save processed data
df.to_csv("processed_transactions.csv", index=False)

print("\nProcessed dataset saved.")
