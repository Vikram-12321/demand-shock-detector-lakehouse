-- 02_create_silver_tables.sql
-- Cleaned layer: typed, deduplicated, non-negative demand values.

USE CATALOG demand_shock_cat;
USE SCHEMA retail_ds;

CREATE TABLE IF NOT EXISTS demand_shock_cat.retail_ds.silver_demand_clean (
  date      DATE    NOT NULL COMMENT 'Calendar date',
  store_id  STRING  NOT NULL COMMENT 'Store identifier',
  item_id   STRING  NOT NULL COMMENT 'Item / SKU identifier',
  demand    DOUBLE  NOT NULL COMMENT 'Cleaned non-negative demand (DOUBLE)',
  row_hash  STRING           COMMENT 'SHA-256 of store_id || item_id || date || demand for deduplication'
)
USING DELTA
COMMENT 'Silver layer: cleaned, typed and deduplicated demand data'
TBLPROPERTIES (
  'delta.autoOptimize.optimizeWrite' = 'true',
  'delta.autoOptimize.autoCompact'   = 'true'
);
