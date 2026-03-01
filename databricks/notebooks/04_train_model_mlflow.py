# Databricks notebook source
# 04_train_model_mlflow.py — Train a demand forecasting model and log to MLflow.

# COMMAND ----------

import sys
import os
import json
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
from 00_config import (  # noqa: F401, E402
    GOLD_TABLE,
    FEATURE_COLS,
    LABEL_COL,
    TRAIN_START,
    TRAIN_END,
    VALID_END,
    RANDOM_SEED,
    MLFLOW_MODEL_NAME,
)

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# COMMAND ----------

ARTIFACT_PATH = "random_forest_demand"


# COMMAND ----------


def load_gold_as_pandas(spark, table_name: str) -> pd.DataFrame:
    """Load the gold feature table into a Pandas DataFrame.

    Drops rows where any feature column or the label is null.
    """
    df = spark.read.table(table_name).toPandas()
    df = df.dropna(subset=FEATURE_COLS + [LABEL_COL])
    df["date"] = pd.to_datetime(df["date"])
    return df


# COMMAND ----------


def split_train_valid(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based (not random) train / validation split.

    Returns (train_df, valid_df).
    """
    train = df[(df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END)].copy()
    valid = df[(df["date"] > TRAIN_END)  & (df["date"] <= VALID_END)].copy()
    print(f"Train rows: {len(train):,} | Valid rows: {len(valid):,}")
    return train, valid


# COMMAND ----------


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """Fit a RandomForestRegressor on the training data."""
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


# COMMAND ----------


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """Compute MAE and RMSE metrics."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true.values - y_pred) ** 2))
    return {"mae": round(float(mae), 4), "rmse": round(float(rmse), 4)}


# COMMAND ----------


def log_mlflow_run(
    params: dict,
    metrics: dict,
    model: Any,
    artifact_path: str,
) -> str:
    """Log parameters, metrics, model and extra artifacts to MLflow.

    Returns the run_id string.
    """
    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path=artifact_path)

        # Feature importance CSV
        if hasattr(model, "feature_importances_"):
            fi_df = pd.DataFrame(
                {"feature": FEATURE_COLS, "importance": model.feature_importances_}
            ).sort_values("importance", ascending=False)
            fi_path = "/tmp/feature_importance.csv"
            fi_df.to_csv(fi_path, index=False)
            mlflow.log_artifact(fi_path, artifact_path="reports")

        # Training summary JSON
        summary = {"params": params, "metrics": metrics}
        summary_path = "/tmp/training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        mlflow.log_artifact(summary_path, artifact_path="reports")

        run_id = run.info.run_id
    print(f"[mlflow] Logged run {run_id}")
    return run_id


# COMMAND ----------


def main() -> None:
    from pyspark.sql import SparkSession  # noqa: PLC0415

    spark = SparkSession.builder.getOrCreate()

    # Enable MLflow autologging (optional, belt-and-suspenders)
    mlflow.sklearn.autolog(disable=True)

    print(f"[train] Loading gold table: {GOLD_TABLE}")
    df = load_gold_as_pandas(spark, GOLD_TABLE)

    train_df, valid_df = split_train_valid(df)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[LABEL_COL]
    X_valid = valid_df[FEATURE_COLS]
    y_valid = valid_df[LABEL_COL]

    print("[train] Training RandomForestRegressor …")
    model = train_model(X_train, y_train)

    y_pred = model.predict(X_valid)
    metrics = evaluate(y_valid, y_pred)
    print(f"[train] Validation metrics: {metrics}")

    params = {
        "model_type": type(model).__name__,
        "features":   ",".join(FEATURE_COLS),
        **{k: v for k, v in model.get_params().items() if k in ("n_estimators", "max_depth", "random_state")},
    }

    run_id = log_mlflow_run(params, metrics, model, ARTIFACT_PATH)
    print(f"[train] run_id = {run_id}")

    # Persist run_id for downstream notebook
    dbutils.jobs.taskValues.set(key="run_id", value=run_id)  # noqa: F821


if __name__ == "__main__":
    main()
