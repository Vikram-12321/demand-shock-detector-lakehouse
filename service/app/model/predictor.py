"""predictor.py — Run inference using a loaded MLflow model."""

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Feature order must match training
FEATURE_ORDER = [
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "rolling_std_7",
    "dow",
    "weekofyear",
    "month",
]


def predict_one(model: Any, features: dict) -> float:
    """Run model inference for a single observation.

    Args:
        model:    An MLflow pyfunc model (or sklearn estimator).
        features: Dict mapping feature name → value.

    Returns:
        Predicted next-day demand as a float.
    """
    row = pd.DataFrame([[features[col] for col in FEATURE_ORDER]], columns=FEATURE_ORDER)
    prediction = model.predict(row)
    # Handle numpy array or scalar
    result = float(prediction[0]) if hasattr(prediction, "__len__") else float(prediction)
    logger.debug("predict_one result: %.4f", result)
    return result
