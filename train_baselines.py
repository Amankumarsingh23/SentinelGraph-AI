
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("outputs/models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load our chronological splits
train = pd.read_csv(DATA_DIR / "train.csv")
val = pd.read_csv(DATA_DIR / "validation.csv")

# Use the most recent part of the training period
# for faster initial experimentation.
train = train.tail(100_000).copy()

numerical = ["TransactionAmt", "TransactionDT"]
categorical = ["ProductCD", "card4", "card6"]
features = numerical + categorical

X_train = train[features].copy()
y_train = train["isFraud"]

X_val = val[features].copy()
y_val = val["isFraud"]

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))

# 2. Missing-value handling for categorical columns
for column in categorical:
    X_train[column] = (
        X_train[column].fillna("Missing").astype(str)
    )
    X_val[column] = (
        X_val[column].fillna("Missing").astype(str)
    )

# 3. Preprocessing for scikit-learn models
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("encoder", OneHotEncoder(
        handle_unknown="ignore"
    ))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical),
    ("cat", categorical_pipeline, categorical)
])

models = {
    "Majority": DummyClassifier(
        strategy="prior"
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42
    )
}

results = []

# 4. Train scikit-learn baselines
for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    print(f"\nTraining {name}...")
    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_val)[:, 1]

    ap = average_precision_score(
        y_val, probabilities
    )

    print(f"{name} Average Precision: {ap:.4f}")
    results.append((name, ap))

# 5. LightGBM and CatBoost use their
# own categorical-feature handling.
X_train_tree = X_train.copy()
X_val_tree = X_val.copy()

# LightGBM: align categorical levels using
# training categories only.
for column in categorical:
    categories = pd.Index(
        X_train_tree[column].unique()
    )
    dtype = pd.CategoricalDtype(
        categories=categories
    )
    X_train_tree[column] = X_train_tree[column].astype(dtype)
    X_val_tree[column] = X_val_tree[column].astype(dtype)

print("\nTraining LightGBM...")

lgbm = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    verbosity=-1,
    n_jobs=-1
)

lgbm.fit(X_train_tree, y_train)

lgbm_prob = lgbm.predict_proba(X_val_tree)[:, 1]

lgbm_ap = average_precision_score(
    y_val, lgbm_prob
)
results.append(("LightGBM", lgbm_ap))

print(f"LightGBM Average Precision: {lgbm_ap:.4f}")

# CatBoost can use the original string categories.
print("\nTraining CatBoost...")

catboost = CatBoostClassifier(
    iterations=200,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    verbose=False,
    random_seed=42,
    train_dir=str(OUTPUT_DIR / "catboost_info"),
    thread_count=-1
)

catboost.fit(
    X_train,
    y_train,
    cat_features=categorical
)

cat_prob = catboost.predict_proba(X_val)[:, 1]

cat_ap = average_precision_score(
    y_val, cat_prob
)
results.append(("CatBoost", cat_ap))

print(f"CatBoost Average Precision: {cat_ap:.4f}")

# 6. Compare results
summary = pd.DataFrame(
    results,
    columns=["Model", "Average_Precision"]
).sort_values(
    "Average_Precision", ascending=False
)

print("\nMODEL COMPARISON")
print(summary.to_string(index=False))

summary.to_csv(
    OUTPUT_DIR / "baseline_comparison.csv",
    index=False
)

print("\nBaseline training completed!")
