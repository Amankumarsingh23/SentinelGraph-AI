"""Local precomputed-feature scoring; no historical-state updates or alert policy."""

from contextlib import asynccontextmanager
import math

from catboost import CatBoostError
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.model import (
    CONTRACT_VERSION, EXPECTED_MODEL_SHA256, FEATURE_COLUMNS, FEATURE_MAPPING,
    MODEL_NAME, MODEL_PATH, PREPROCESSING, feature_frame, load_model,
)
from api.schemas import HealthResponse, ModelInfoResponse, PredictionResponse, TransactionRequest


def create_app(model_path=MODEL_PATH, expected_sha256=EXPECTED_MODEL_SHA256):
    @asynccontextmanager
    async def lifespan(application):
        application.state.serving_model = load_model(model_path, expected_sha256)
        try:
            yield
        finally:
            application.state.serving_model = None

    application = FastAPI(
        title="SentinelGraph AI", version=CONTRACT_VERSION, lifespan=lifespan,
        description=("Local development scoring using precomputed historical features. "
                     "Returns an uncalibrated score without assigning a fraud decision."),
    )
    application.state.serving_model = None

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError):
        # Raw NaN/Infinity values cannot be JSON-encoded in an error response.
        # Keep useful locations/messages without echoing transaction contents.
        details = [{key: item[key] for key in ("type", "loc", "msg")} for item in error.errors()]
        return JSONResponse(status_code=422, content={"detail": details})

    def ready_model():
        model = application.state.serving_model
        if model is None:
            raise HTTPException(status_code=503, detail="Model is not ready")
        return model

    @application.get("/health", response_model=HealthResponse)
    def health():
        model = ready_model()
        return HealthResponse(model_loaded=True, model_name=MODEL_NAME, model_sha256=model.sha256)

    @application.get("/model-info", response_model=ModelInfoResponse)
    def model_info():
        model = ready_model()
        return ModelInfoResponse(
            model_name=MODEL_NAME, model_sha256=model.sha256,
            feature_contract_version=CONTRACT_VERSION, feature_columns=FEATURE_COLUMNS,
            request_to_model_features=FEATURE_MAPPING, preprocessing=PREPROCESSING,
            missing_value_policy="omitted or null history -> NaN; zero remains zero; amount is required",
            score_type="uncalibrated_model_score", decision="not_assigned",
            evaluation_scope="reused development validation; final model selection pending",
            historical_features="client-supplied precomputed values; no trusted online feature service integrated",
        )

    @application.post("/predict", response_model=PredictionResponse)
    def predict(transaction: TransactionRequest):
        model = ready_model()
        try:
            score = float(model.classifier.predict_proba(feature_frame(transaction), thread_count=2)[0, 1])
        except (CatBoostError, ValueError, OverflowError):
            raise HTTPException(status_code=500, detail="Model inference failed") from None
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise HTTPException(status_code=500, detail="Model returned an invalid score")
        return PredictionResponse(transaction_id=transaction.transaction_id, risk_score=score,
                                  model_sha256=model.sha256)

    return application


app = create_app()
