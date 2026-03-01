# Databricks notebook source
# 02_silver_clean.py — Clean bronze → silver Delta table.

# COMMAND ----------

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from 00_config import BRONZE_TABLE, SILVER_TABLE  # noqa: F401, E402

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    col,
    concat_ws,
    sha2,
    to_date,
)


# COMMAND ----------


def clean_bronze_to_silver(df: DataFrame) -> DataFrame:
    """Apply cleaning rules to a bronze DataFrame and return a silver DataFrame.

    Transformations:
    - Cast `date` to DATE
    - Cast `demand` to DOUBLE
    - Filter demand >= 0
    - Drop rows with null store_id / item_id / date
    - Add row_hash = sha2(store_id || item_id || date || demand, 256)
    - Deduplicate on row_hash
    """
    df_clean = (
        df
        .withColumn("date",   to_date(col("date"), "yyyy-MM-dd"))
        .withColumn("demand", col("demand").cast("double"))
        .filter(col("demand") >= 0)
        .filter(col("store_id").isNotNull())
        .filter(col("item_id").isNotNull())
        .filter(col("date").isNotNull())
        .withColumn(
            "row_hash",
            sha2(concat_ws("||", col("store_id"), col("item_id"), col("date").cast("string"), col("demand").cast("string")), 256),
        )
        .dropDuplicates(["row_hash"])
        .select("date", "store_id", "item_id", "demand", "row_hash")
    )
    return df_clean


# COMMAND ----------


def write_silver(df: DataFrame, table_name: str) -> None:
    """Overwrite the silver Delta table with the cleaned DataFrame."""
    (
        df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(table_name)
    )
    print(f"[silver] Wrote {df.count():,} rows → {table_name}")


# COMMAND ----------


def main() -> None:
    spark = SparkSession.builder.getOrCreate()

    print(f"[silver] Reading from {BRONZE_TABLE}")
    df_bronze = spark.read.table(BRONZE_TABLE)

    df_silver = clean_bronze_to_silver(df_bronze)
    df_silver.printSchema()
    df_silver.show(5, truncate=False)

    print(f"[silver] Writing to {SILVER_TABLE}")
    write_silver(df_silver, SILVER_TABLE)
    print("[silver] Done.")


if __name__ == "__main__":
    main()
