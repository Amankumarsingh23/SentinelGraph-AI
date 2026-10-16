"""Strictly earlier-time histories; card1 is a proxy, not a verified customer ID."""
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data/processed"
FEATURES = ["card1_prior_count", "card1_prior_mean_amount",
            "card1_time_since_previous", "card1_amount_ratio", "card1_missing"]


def historical_features(frame):
    """Preserve row order; same-time peers cannot observe one another.

    Missing/nonfinite amounts count as transactions but not observed amounts.
    Missing/nonfinite entities have no history. Invalid times are rejected.
    Earlier observed inputs remain available across splits; labels are not used.
    """
    times = pd.to_numeric(frame["TransactionDT"], errors="raise")
    if not np.isfinite(times.to_numpy(dtype=float)).all():
        raise ValueError("TransactionDT must contain finite, nonmissing times")
    entity = frame["card1"].replace([np.inf, -np.inf], np.nan)
    amount = frame["TransactionAmt"].replace([np.inf, -np.inf], np.nan)
    events = pd.DataFrame({"card1": entity.to_numpy(),
                           "TransactionDT": times.to_numpy(),
                           "amount": amount.to_numpy(), "row": np.arange(len(frame))})
    # Shift whole timestamp batches: neither current nor simultaneous events
    # enter historical counts, observed-amount sums/counts, or previous times.
    batches = events.groupby(["card1", "TransactionDT"], sort=True).agg(
        transactions=("row", "size"), observed=("amount", "count"),
        amount_sum=("amount", "sum"),
    ).reset_index()
    cumulative = batches.groupby("card1", sort=False)[
        ["transactions", "observed", "amount_sum"]
    ].cumsum()
    prior = cumulative.groupby(batches["card1"], sort=False).shift(fill_value=0)
    batches["card1_prior_count"] = prior["transactions"]
    batches["card1_prior_mean_amount"] = (
        prior["amount_sum"] / prior["observed"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    batches["card1_time_since_previous"] = batches.groupby("card1")["TransactionDT"].diff()
    joined = events.merge(batches[["card1", "TransactionDT"] + FEATURES[:3]],
                          on=["card1", "TransactionDT"], how="left",
                          validate="many_to_one").sort_values("row")
    output = joined[FEATURES[:3]].copy()
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        output["card1_amount_ratio"] = (
            joined["amount"] / joined["card1_prior_mean_amount"].replace(0, np.nan)
        )
    output["card1_missing"] = joined["card1"].isna().astype(int)
    output = output.replace([np.inf, -np.inf], np.nan)
    output.index = frame.index
    return output[FEATURES]


def main():
    parts = []
    for split in ["train", "validation", "test"]:
        frame = pd.read_csv(DATA_DIR / f"{split}.csv")
        parts.append(pd.concat([
            frame, pd.DataFrame({"_split": split}, index=frame.index)
        ], axis=1))
    df = pd.concat(parts, ignore_index=True).sort_values(
        ["TransactionDT", "TransactionID"], kind="stable"
    ).reset_index(drop=True)
    df = pd.concat([df, historical_features(df)], axis=1)
    for split in ["train", "validation", "test"]:
        subset = df.loc[df["_split"] == split].drop(columns="_split")
        subset.to_csv(DATA_DIR / f"{split}_features.csv", index=False)
        print(f"\n{split.upper()} FEATURE REPORT\nRows: {len(subset)}")
        print(subset[FEATURES].describe())
    print("\nFeature engineering completed!")


if __name__ == "__main__":
    main()
