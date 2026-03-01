# Databricks notebook source
# 05_batch_scoring_writeback.py — Score latest data and write predictions to gold table.

# COMMAND ----------

import sys
import os
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
from 00_config import (  # noqa: F401, E402
    GOLD_TABLE,
    PRED_TABLE,
    FEATURE_COLS,
    MLFLOW_MODEL_NAME,
)

import mlflow
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


# COMMAND ----------


def load_registered_model(
    model_name: str,
    model_version: str | None = None,
) -> tuple[Any, str]:
    """Load a registered MLflow model.

    If model_version is None, loads the Champion alias (or latest version).
    Returns (model, version_string).
    """
    if model_version:
        uri = f"models:/{model_name}/{model_version}"
    else:
        # Try Champion alias first; fall back to latest version
        try:
            uri = f"models:/{model_name}@Champion"
            model = mlflow.pyfunc.load_model(uri)
            client = mlflow.tracking.MlflowClient()
            mv = client.get_model_version_by_alias(model_name, "Champion")
            return model, mv.version
        except Exception:
            client = mlflow.tracking.MlflowClient()
            versions = client.get_latest_versions(model_name)
            if not versions:
                raise ValueError(f"No versions found for model '{model_name}'")
            latest = versions[-1]
            uri = f"models:/{model_name}/{latest.version}"
            model = mlflow.pyfunc.load_model(uri)
            return model, latest.version

    model = mlflow.pyfunc.load_model(uri)
    return model, model_version


# COMMAND ----------


def score(
    df_gold: DataFrame,
    model: Any,
    feature_cols: list[str],
    model_name: str = MLFLOW_MODEL_NAME,
    model_version: str = "unknown",
) -> DataFrame:
    """Score the gold feature DataFrame with the given MLflow model.

    Adds: y_hat, shock_score, is_shock_pred, model_name, model_version, scored_at.

    Shock score formula (batch — actual demand is available):
        shock_score = abs(demand - y_hat) / (rolling_std_7 + 1e-6)
    This differs from the real-time API where no actual demand is known and
    `abs(y_hat - rolling_mean_7)` is used instead.
    """
    import pandas as pd  # noqa: PLC0415

    # Score via pandas for simplicity; use pandas_udf for very large tables
    X_pd = df_gold.select(*feature_cols).toPandas()
    y_hat_arr = model.predict(X_pd)

    spark = SparkSession.builder.getOrCreate()
    df_scores_pd = df_gold.select("date", "store_id", "item_id", "demand", "rolling_std_7").toPandas()
    df_scores_pd["y_hat"] = y_hat_arr
    df_scores_pd["shock_score"] = (
        (df_scores_pd["demand"] - df_scores_pd["y_hat"]).abs()
        / (df_scores_pd["rolling_std_7"].fillna(0) + 1e-6)
    )
    df_scores_pd["is_shock_pred"] = (df_scores_pd["shock_score"] >= 3.0).astype(int)
    df_scores_pd["model_name"]    = model_name
    df_scores_pd["model_version"] = model_version

    df_out = spark.createDataFrame(df_scores_pd[
        ["date", "store_id", "item_id", "y_hat", "shock_score", "is_shock_pred",
         "model_name", "model_version"]
    ])
    df_out = df_out.withColumn("scored_at", F.current_timestamp())
    return df_out


# COMMAND ----------


def main() -> None:
    spark = SparkSession.builder.getOrCreate()

    # Retrieve run_id from upstream task (if available)
    try:
        model_version = dbutils.jobs.taskValues.get(  # noqa: F821
            taskKey="04_train_model_mlflow", key="run_id"
        )
    except Exception:
        model_version = None

    print(f"[score] Loading model '{MLFLOW_MODEL_NAME}' version={model_version or 'Champion'}")
    model, version = load_registered_model(MLFLOW_MODEL_NAME, model_version)
    print(f"[score] Loaded model version {version}")

    print(f"[score] Reading gold features from {GOLD_TABLE}")
    df_gold = spark.read.table(GOLD_TABLE).dropna(subset=FEATURE_COLS)

    print("[score] Scoring …")
    df_pred = score(df_gold, model, FEATURE_COLS, model_name=MLFLOW_MODEL_NAME, model_version=version)
    df_pred.show(5, truncate=False)

    print(f"[score] Appending {df_pred.count():,} rows → {PRED_TABLE}")
    (
        df_pred.write
        .format("delta")
        .mode("append")
        .saveAsTable(PRED_TABLE)
    )
    print("[score] Done.")


if __name__ == "__main__":
    main()
