# Databricks notebook source
# 01_bronze_ingest.py — Read CSV from DBFS and append into the bronze Delta table.

# COMMAND ----------

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from 00_config import BRONZE_TABLE, SAMPLE_PATH  # noqa: F401, E402

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import current_timestamp, input_file_name


# COMMAND ----------


def read_raw_demand_csv(path: str) -> DataFrame:
    """Read raw demand CSV from DBFS (or local path) into a Spark DataFrame.

    Expected columns: date, store_id, item_id, demand
    Returns the DataFrame with two extra provenance columns added.
    """
    spark = SparkSession.builder.getOrCreate()
    df = (
        spark.read.option("header", "true")
        .option("inferSchema", "false")   # keep everything as STRING for bronze
        .csv(path)
        .withColumn("ingested_at", current_timestamp())
        .withColumn("source_file", input_file_name())
    )
    return df


# COMMAND ----------


def write_bronze(df: DataFrame, table_name: str) -> None:
    """Append the DataFrame into the bronze Delta table."""
    (
        df.write
        .format("delta")
        .mode("append")
        .option("mergeSchema", "true")
        .saveAsTable(table_name)
    )
    print(f"[bronze] Appended {df.count():,} rows → {table_name}")


# COMMAND ----------


def main() -> None:
    print(f"[bronze] Reading from {SAMPLE_PATH}")
    df = read_raw_demand_csv(SAMPLE_PATH)
    df.printSchema()
    df.show(5, truncate=False)

    print(f"[bronze] Writing to {BRONZE_TABLE}")
    write_bronze(df, BRONZE_TABLE)
    print("[bronze] Done.")


if __name__ == "__main__":
    main()
