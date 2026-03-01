-- 01_create_bronze_tables.sql
-- Raw ingestion layer: append-only Delta table with provenance columns.

USE CATALOG demand_shock_cat;
USE SCHEMA retail_ds;

CREATE TABLE IF NOT EXISTS demand_shock_cat.retail_ds.bronze_demand_raw (
  date         STRING   COMMENT 'Raw date string from source CSV',
  store_id     STRING   COMMENT 'Store identifier',
  item_id      STRING   COMMENT 'Item / SKU identifier',
  demand       STRING   COMMENT 'Raw demand value (may require casting)',
  ingested_at  TIMESTAMP DEFAULT current_timestamp() COMMENT 'Timestamp when row was ingested',
  source_file  STRING   COMMENT 'Source file path (input_file_name())'
)
USING DELTA
COMMENT 'Bronze layer: raw demand data ingested from CSV files'
TBLPROPERTIES (
  'delta.autoOptimize.optimizeWrite' = 'true',
  'delta.autoOptimize.autoCompact'   = 'true'
);
