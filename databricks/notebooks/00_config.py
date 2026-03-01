# Databricks notebook source
# 00_config.py — Shared constants for the Demand Shock Detector pipeline.
# All other notebooks import from this file (or re-define inline if needed).

# COMMAND ----------

# Unity Catalog identifiers
CATALOG = "demand_shock_cat"
SCHEMA = "retail_ds"

BRONZE_TABLE = f"{CATALOG}.{SCHEMA}.bronze_demand_raw"
SILVER_TABLE = f"{CATALOG}.{SCHEMA}.silver_demand_clean"
GOLD_TABLE   = f"{CATALOG}.{SCHEMA}.gold_demand_features"
PRED_TABLE   = f"{CATALOG}.{SCHEMA}.gold_predictions_daily"

# DBFS paths
RAW_MOUNT_PATH = "dbfs:/FileStore/demand_shock/raw/"
SAMPLE_PATH    = "dbfs:/FileStore/demand_shock/sample/demand_sample.csv"

# Reproducibility
RANDOM_SEED = 42

# Training window
TRAIN_START = "2014-01-01"
TRAIN_END   = "2015-12-31"
VALID_END   = "2016-06-30"

# Feature columns expected by the model
FEATURE_COLS = [
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "rolling_std_7",
    "dow",
    "weekofyear",
    "month",
]
LABEL_COL = "target_next_day"

# MLflow
MLFLOW_MODEL_NAME = "demand_shock_detector"


def get_spark():
    """Return the active Spark session (works inside Databricks and local tests)."""
    from pyspark.sql import SparkSession  # noqa: PLC0415

    return SparkSession.builder.getOrCreate()


if __name__ == "__main__":
    print("Config loaded:")
    print(f"  CATALOG       = {CATALOG}")
    print(f"  BRONZE_TABLE  = {BRONZE_TABLE}")
    print(f"  SILVER_TABLE  = {SILVER_TABLE}")
    print(f"  GOLD_TABLE    = {GOLD_TABLE}")
    print(f"  PRED_TABLE    = {PRED_TABLE}")
