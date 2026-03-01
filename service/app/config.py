"""config.py — Runtime configuration loaded from environment variables."""

import os

# MLflow settings
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME: str = os.getenv("MODEL_NAME", "demand_shock_detector")
MODEL_VERSION: str | None = os.getenv("MODEL_VERSION") or None  # None → Champion / latest

# Shock detection threshold
SHOCK_THRESHOLD: float = 3.0
