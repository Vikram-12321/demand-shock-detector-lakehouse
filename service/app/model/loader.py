"""loader.py — Load MLflow models for serving."""

import logging
from typing import Any

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


def load_model_from_mlflow(model_uri: str) -> Any:
    """Load any MLflow model from a URI string.

    Examples:
        load_model_from_mlflow("models:/demand_shock_detector/1")
        load_model_from_mlflow("models:/demand_shock_detector@Champion")
        load_model_from_mlflow("runs:/<run_id>/random_forest_demand")
    """
    logger.info("Loading model from URI: %s", model_uri)
    return mlflow.pyfunc.load_model(model_uri)


def load_champion_model(model_name: str) -> tuple[Any, str]:
    """Load the Champion (or latest) version of a registered model.

    Returns (model, version_string).
    """
    client = MlflowClient()

    # 1. Try Champion alias (Unity Catalog model registry)
    try:
        uri = f"models:/{model_name}@Champion"
        model = load_model_from_mlflow(uri)
        mv = client.get_model_version_by_alias(model_name, "Champion")
        logger.info("Loaded model '%s' version %s via Champion alias", model_name, mv.version)
        return model, str(mv.version)
    except Exception as alias_err:
        logger.debug("Champion alias not found (%s); falling back to latest version", alias_err)

    # 2. Fall back to the latest version available in any stage
    versions = client.get_latest_versions(model_name)
    if not versions:
        raise ValueError(f"No versions found for registered model '{model_name}'")

    # Pick the version with the highest version number
    latest = max(versions, key=lambda v: int(v.version))
    uri = f"models:/{model_name}/{latest.version}"
    model = load_model_from_mlflow(uri)
    logger.info("Loaded model '%s' latest version %s", model_name, latest.version)
    return model, str(latest.version)
