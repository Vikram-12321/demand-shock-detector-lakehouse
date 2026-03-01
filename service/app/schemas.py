"""schemas.py — Pydantic request / response models for the API."""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    lag_1: float = Field(..., description="Demand 1 day prior")
    lag_7: float = Field(..., description="Demand 7 days prior")
    rolling_mean_7: float = Field(..., description="7-day rolling average demand")
    rolling_std_7: float = Field(..., ge=0.0, description="7-day rolling std dev of demand")
    dow: int = Field(..., ge=1, le=7, description="Day of week (1=Sunday, 7=Saturday)")
    weekofyear: int = Field(..., ge=1, le=53, description="ISO week of year")
    month: int = Field(..., ge=1, le=12, description="Calendar month (1-12)")


class PredictResponse(BaseModel):
    y_hat: float = Field(..., description="Predicted next-day demand")
    shock_score: float = Field(..., description="abs(y_hat - rolling_mean_7) / (rolling_std_7 + 1e-6)")
    is_shock_pred: int = Field(..., description="1 if shock_score >= 3.0, else 0")
    model_name: str = Field(..., description="MLflow registered model name")
    model_version: str = Field(..., description="MLflow model version used")
