-- 04_grants.sql
-- Grant read access to account-level users / groups.
-- Replace <your-user@domain.com> or <your-group-name> with your actual principal.
-- If running in a single-user workspace, grant to yourself.

USE CATALOG demand_shock_cat;

-- Catalog-level usage
GRANT USAGE ON CATALOG demand_shock_cat TO `account users`;

-- Schema-level usage
GRANT USAGE ON SCHEMA demand_shock_cat.retail_ds TO `account users`;

-- Table-level SELECT grants
GRANT SELECT ON TABLE demand_shock_cat.retail_ds.bronze_demand_raw      TO `account users`;
GRANT SELECT ON TABLE demand_shock_cat.retail_ds.silver_demand_clean     TO `account users`;
GRANT SELECT ON TABLE demand_shock_cat.retail_ds.gold_demand_features    TO `account users`;
GRANT SELECT ON TABLE demand_shock_cat.retail_ds.gold_predictions_daily  TO `account users`;

-- NOTE: If your workspace does not have an "account users" group,
-- grant to a specific user instead, e.g.:
--   GRANT USAGE ON CATALOG demand_shock_cat TO `user@example.com`;
