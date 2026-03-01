-- 03_create_gold_tables.sql
-- Feature-engineered layer: ML-ready features + prediction output table.

USE CATALOG demand_shock_cat;
USE SCHEMA retail_ds;

-- Gold feature table
CREATE TABLE IF NOT EXISTS demand_shock_cat.retail_ds.gold_demand_features (
  date              DATE    NOT NULL COMMENT 'Calendar date',
  store_id          STRING  NOT NULL COMMENT 'Store identifier',
  item_id           STRING  NOT NULL COMMENT 'Item / SKU identifier',
  demand            DOUBLE           COMMENT 'Actual demand for this row',
  lag_1             DOUBLE           COMMENT 'Demand 1 day prior (same store/item)',
  lag_7             DOUBLE           COMMENT 'Demand 7 days prior (same store/item)',
  rolling_mean_7    DOUBLE           COMMENT '7-day rolling average demand',
  rolling_std_7     DOUBLE           COMMENT '7-day rolling standard deviation of demand',
  dow               INT              COMMENT 'Day of week (1=Sunday … 7=Saturday in Spark)',
  weekofyear        INT              COMMENT 'ISO week of year',
  month             INT              COMMENT 'Calendar month (1-12)',
  target_next_day   DOUBLE           COMMENT 'Lead(demand, 1) — label for next-day prediction',
  is_shock_label    INT              COMMENT '1 if abs_z >= 3.0 (demand shock), else 0'
)
USING DELTA
COMMENT 'Gold layer: ML-ready features for demand shock detection'
TBLPROPERTIES (
  'delta.autoOptimize.optimizeWrite' = 'true',
  'delta.autoOptimize.autoCompact'   = 'true'
);

-- Predictions output table
CREATE TABLE IF NOT EXISTS demand_shock_cat.retail_ds.gold_predictions_daily (
  date           DATE      NOT NULL COMMENT 'Prediction date',
  store_id       STRING    NOT NULL COMMENT 'Store identifier',
  item_id        STRING    NOT NULL COMMENT 'Item / SKU identifier',
  y_hat          DOUBLE             COMMENT 'Predicted next-day demand',
  shock_score    DOUBLE             COMMENT 'abs(demand - y_hat) / (rolling_std_7 + 1e-6)',
  is_shock_pred  INT                COMMENT '1 if shock_score >= 3.0, else 0',
  model_name     STRING             COMMENT 'MLflow registered model name',
  model_version  STRING             COMMENT 'MLflow model version used for scoring',
  scored_at      TIMESTAMP          COMMENT 'Timestamp when scoring ran'
)
USING DELTA
COMMENT 'Gold layer: daily demand predictions with shock flags'
TBLPROPERTIES (
  'delta.autoOptimize.optimizeWrite' = 'true',
  'delta.autoOptimize.autoCompact'   = 'true'
);
