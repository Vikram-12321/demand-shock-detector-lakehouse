-- 00_create_catalog_schema.sql
-- Create the Unity Catalog catalog and schema for the Demand Shock Detector project.

CREATE CATALOG IF NOT EXISTS demand_shock_cat
  COMMENT 'Demand Shock Detector — top-level Unity Catalog catalog';

CREATE SCHEMA IF NOT EXISTS demand_shock_cat.retail_ds
  COMMENT 'Retail demand dataset: bronze / silver / gold layers';

USE CATALOG demand_shock_cat;
USE SCHEMA retail_ds;
