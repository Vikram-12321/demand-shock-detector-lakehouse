"""main.py — FastAPI application for the Demand Shock Detector inference service."""

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import mlflow
from fastapi import FastAPI, HTTPException

from app.config import MLFLOW_TRACKING_URI, MODEL_NAME, MODEL_VERSION, SHOCK_THRESHOLD
from app.features.transform import compute_shock_score
from app.logging_config import configure_logging
from app.model.loader import load_champion_model, load_model_from_mlflow
from app.model.predictor import predict_one
from app.schemas import PredictRequest, PredictResponse
from app.utils.health import build_health_response

configure_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------
_model: Any = None
_model_version: str = "unknown"
_loaded_at: str = "never"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the MLflow model once at startup."""
    global _model, _model_version, _loaded_at

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info("MLflow tracking URI: %s", MLFLOW_TRACKING_URI)

    try:
        if MODEL_VERSION:
            uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
            _model = load_model_from_mlflow(uri)
            _model_version = MODEL_VERSION
        else:
            _model, _model_version = load_champion_model(MODEL_NAME)
        _loaded_at = datetime.now(timezone.utc).isoformat()
        logger.info("Model loaded: %s v%s at %s", MODEL_NAME, _model_version, _loaded_at)
    except Exception as exc:
        logger.warning("Could not load model at startup: %s", exc)
        logger.warning("Service will start but /predict will return 503 until a model is loaded.")

    yield


app = FastAPI(
    title="Demand Shock Detector API",
    description="Real-time demand shock detection using MLflow-registered model",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", tags=["ops"])
def health():
    """Liveness probe — returns {status: ok}."""
    return build_health_response()


@app.get("/model-info", tags=["ops"])
def model_info():
    """Return metadata about the currently loaded model."""
    return {
        "model_name": MODEL_NAME,
        "model_version": _model_version,
        "loaded_at": _loaded_at,
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
    }


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(request: PredictRequest) -> PredictResponse:
    """Predict next-day demand and compute shock score.

    Body fields map directly to the gold feature table columns.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check MLflow configuration.")

    features = request.model_dump()
    y_hat = predict_one(_model, features)

    shock_score = compute_shock_score(
        rolling_mean_7=request.rolling_mean_7,
        rolling_std_7=request.rolling_std_7,
        y_hat=y_hat,
    )
    is_shock_pred = 1 if shock_score >= SHOCK_THRESHOLD else 0

    return PredictResponse(
        y_hat=y_hat,
        shock_score=shock_score,
        is_shock_pred=is_shock_pred,
        model_name=MODEL_NAME,
        model_version=_model_version,
    )
