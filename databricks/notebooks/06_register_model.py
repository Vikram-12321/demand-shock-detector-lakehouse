# Databricks notebook source
# 06_register_model.py — Register the best MLflow run into the Model Registry.

# COMMAND ----------

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from 00_config import MLFLOW_MODEL_NAME  # noqa: F401, E402

import mlflow
from mlflow.tracking import MlflowClient


# COMMAND ----------

ARTIFACT_PATH = "random_forest_demand"


# COMMAND ----------


def register_best_run(
    model_name: str,
    run_id: str,
    artifact_path: str,
) -> tuple[str, str]:
    """Register the model from run_id into the MLflow Model Registry.

    Sets the 'Champion' alias on the newly registered version.
    Returns (model_name, version_string).
    """
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/{artifact_path}"

    # Create registered model if it doesn't exist
    try:
        client.create_registered_model(
            model_name,
            description="Demand Shock Detector — RandomForest next-day demand forecast",
        )
        print(f"[register] Created registered model '{model_name}'")
    except mlflow.exceptions.MlflowException:
        print(f"[register] Registered model '{model_name}' already exists")

    # Register new version
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = mv.version
    print(f"[register] Registered version {version} from run {run_id}")

    # Set 'Champion' alias (Unity Catalog model registry)
    try:
        client.set_registered_model_alias(model_name, "Champion", version)
        print(f"[register] Set alias 'Champion' → version {version}")
    except Exception as e:
        print(f"[register] Could not set alias (non-UC workspace?): {e}")
        # Fall back to transitioning stage to Production
        try:
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production",
                archive_existing_versions=True,
            )
            print(f"[register] Transitioned version {version} → Production")
        except Exception as e2:
            print(f"[register] Stage transition also failed: {e2}")

    return model_name, str(version)


# COMMAND ----------


def main() -> None:
    # Get run_id from the upstream training task
    try:
        run_id = dbutils.jobs.taskValues.get(  # noqa: F821
            taskKey="04_train_model_mlflow", key="run_id"
        )
    except Exception:
        # Fallback: use the most recent MLflow run in the current experiment
        runs = mlflow.search_runs(order_by=["start_time DESC"], max_results=1)
        if runs.empty:
            raise RuntimeError("No MLflow runs found. Run 04_train_model_mlflow first.")
        run_id = runs.iloc[0]["run_id"]

    print(f"[register] Using run_id = {run_id}")
    name, version = register_best_run(MLFLOW_MODEL_NAME, run_id, ARTIFACT_PATH)
    print(f"[register] Model '{name}' version {version} is now Champion.")

    # Pass downstream
    dbutils.jobs.taskValues.set(key="model_version", value=version)  # noqa: F821


if __name__ == "__main__":
    main()
