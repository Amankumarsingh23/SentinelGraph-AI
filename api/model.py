"""Load the pinned historical model and construct its exact numerical inputs."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd

from api.schemas import TransactionRequest

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "outputs/evaluation/session20/AllFive.cbm"
# Reviewed Session 20 binary, also retained in the Session 21 manifest.
# A new model requires an intentional version/hash update and verification.
EXPECTED_MODEL_SHA256 = "4738766c01b72de4b113d922f5236a9f7d979286b2994ff854cce99e7ddd44ed"
MODEL_NAME = "AllFive"
CONTRACT_VERSION = "session22-v1"
FEATURE_MAPPING = {
    "transaction_amount": "TransactionAmt",
    "graph_card_prior_degree": "graph_card_prior_degree",
    "graph_product_prior_degree": "graph_product_prior_degree",
    "graph_pair_prior_count": "graph_card_product_prior_count",
    "graph_card_distinct_products": "graph_card_distinct_products",
}
FEATURE_COLUMNS = list(FEATURE_MAPPING.values())
PREPROCESSING = "raw numerical features; no scaler or log transform; CatBoost native NaN handling"


@dataclass(frozen=True)
class ServingModel:
    classifier: CatBoostClassifier
    sha256: str


def load_model(path=MODEL_PATH, expected_sha256=EXPECTED_MODEL_SHA256):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError("AllFive model is missing; restore the Session 20 artifact before startup")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected_sha256:
        raise RuntimeError("AllFive model SHA-256 differs from the pinned serving artifact")
    classifier = CatBoostClassifier()
    classifier.load_model(str(path))
    if classifier.feature_names_ != FEATURE_COLUMNS:
        raise RuntimeError("Model feature names/order differ from the serving contract")
    if classifier.classes_.tolist() != [0, 1]:
        raise RuntimeError("Model must expose binary class probabilities in [0, 1] order")
    return ServingModel(classifier, digest)


def feature_frame(transaction: TransactionRequest):
    values = [getattr(transaction, field) for field in FEATURE_MAPPING]
    # JSON null/omitted history becomes NaN. Zero remains zero. Inputs have
    # already rejected infinities rather than silently treating them as missing.
    return prepare_features(pd.DataFrame([[np.nan if value is None else value for value in values]],
                                        columns=FEATURE_COLUMNS, dtype=np.float64))


def prepare_features(frame):
    """Shared serving/batch contract from Session 20: raw values, inf -> NaN."""
    return frame.loc[:, FEATURE_COLUMNS].astype(np.float64).replace([np.inf, -np.inf], np.nan)
