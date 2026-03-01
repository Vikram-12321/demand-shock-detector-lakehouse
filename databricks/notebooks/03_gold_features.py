# Databricks notebook source
# 03_gold_features.py — Build ML-ready feature table from silver layer.

# COMMAND ----------

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from 00_config import SILVER_TABLE, GOLD_TABLE  # noqa: F401, E402

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# COMMAND ----------


def make_features(df_silver: DataFrame) -> DataFrame:
    """Engineer features from the silver demand DataFrame.

    Window: partition by (store_id, item_id), order by date.

    Added columns:
      lag_1, lag_7             — lag features
      rolling_mean_7           — 7-day rolling average
      rolling_std_7            — 7-day rolling std dev
      target_next_day          — lead(demand, 1)
      residual_baseline        — demand - rolling_mean_7
      is_shock_label           — 1 if abs_z >= 3.0 (and stats are not null)
      dow, weekofyear, month   — calendar features
    """
    w_lag  = Window.partitionBy("store_id", "item_id").orderBy("date")
    w_roll = (
        Window.partitionBy("store_id", "item_id")
        .orderBy(F.col("date").cast("long"))
        .rowsBetween(-6, 0)
    )

    df_feat = (
        df_silver
        # Lag features
        .withColumn("lag_1",  F.lag("demand", 1).over(w_lag))
        .withColumn("lag_7",  F.lag("demand", 7).over(w_lag))
        # Rolling stats (7-day window ending on current row)
        .withColumn("rolling_mean_7", F.avg("demand").over(w_roll))
        .withColumn("rolling_std_7",  F.stddev("demand").over(w_roll))
        # Target (next-day demand)
        .withColumn("target_next_day", F.lead("demand", 1).over(w_lag))
        # Residual from rolling baseline
        .withColumn(
            "residual_baseline",
            F.col("demand") - F.col("rolling_mean_7"),
        )
        # Shock label
        .withColumn(
            "abs_z",
            F.abs(
                (F.col("demand") - F.col("rolling_mean_7"))
                / (F.col("rolling_std_7") + 1e-6)
            ),
        )
        .withColumn(
            "is_shock_label",
            F.when(
                F.col("rolling_std_7").isNotNull()
                & F.col("rolling_mean_7").isNotNull()
                & (F.col("abs_z") >= 3.0),
                1,
            ).otherwise(0),
        )
        .drop("abs_z", "residual_baseline")
        # Calendar features
        .withColumn("dow",        F.dayofweek("date"))
        .withColumn("weekofyear", F.weekofyear("date"))
        .withColumn("month",      F.month("date"))
    )
    return df_feat


# COMMAND ----------


def write_gold(df: DataFrame, table_name: str) -> None:
    """Overwrite the gold features Delta table."""
    (
        df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(table_name)
    )
    print(f"[gold] Wrote {df.count():,} rows → {table_name}")


# COMMAND ----------


def main() -> None:
    spark = SparkSession.builder.getOrCreate()

    print(f"[gold] Reading from {SILVER_TABLE}")
    df_silver = spark.read.table(SILVER_TABLE)

    df_gold = make_features(df_silver)
    df_gold.printSchema()
    df_gold.show(5, truncate=False)

    print(f"[gold] Writing to {GOLD_TABLE}")
    write_gold(df_gold, GOLD_TABLE)
    print("[gold] Done.")


if __name__ == "__main__":
    main()
