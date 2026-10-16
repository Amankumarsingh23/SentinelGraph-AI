"""External feature contract; optional history preserves missing versus zero."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TransactionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)

    transaction_id: int = Field(ge=0)
    transaction_amount: float = Field(ge=0)
    graph_card_prior_degree: float | None = Field(default=None, ge=0)
    graph_product_prior_degree: float | None = Field(default=None, ge=0)
    graph_pair_prior_count: float | None = Field(default=None, ge=0)
    graph_card_distinct_products: float | None = Field(default=None, ge=0)


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    transaction_id: int
    risk_score: float = Field(ge=0, le=1)
    model_name: Literal["AllFive"] = "AllFive"
    model_sha256: str
    feature_contract_version: Literal["session22-v1"] = "session22-v1"
    score_type: Literal["uncalibrated_model_score"] = "uncalibrated_model_score"
    decision: Literal["not_assigned"] = "not_assigned"


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model_loaded: bool
    model_name: str
    model_sha256: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_sha256: str
    feature_contract_version: str
    feature_columns: list[str]
    request_to_model_features: dict[str, str]
    preprocessing: str
    missing_value_policy: str
    score_type: str
    decision: str
    evaluation_scope: str
    historical_features: str
