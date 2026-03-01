-- 05_sample_queries.sql
-- Six analytics queries for the Demand Shock Detector lakehouse.

USE CATALOG demand_shock_cat;
USE SCHEMA retail_ds;

-- ─────────────────────────────────────────────
-- Query 1: Row counts per layer
-- ─────────────────────────────────────────────
SELECT 'bronze' AS layer, COUNT(*) AS row_count
FROM demand_shock_cat.retail_ds.bronze_demand_raw
UNION ALL
SELECT 'silver', COUNT(*)
FROM demand_shock_cat.retail_ds.silver_demand_clean
UNION ALL
SELECT 'gold_features', COUNT(*)
FROM demand_shock_cat.retail_ds.gold_demand_features
UNION ALL
SELECT 'gold_predictions', COUNT(*)
FROM demand_shock_cat.retail_ds.gold_predictions_daily
ORDER BY layer;

-- ─────────────────────────────────────────────
-- Query 2: Top shocks by absolute deviation from rolling mean
-- ─────────────────────────────────────────────
SELECT
  date,
  store_id,
  item_id,
  demand,
  rolling_mean_7,
  rolling_std_7,
  ABS(demand - rolling_mean_7) / (rolling_std_7 + 1e-6) AS abs_z_score
FROM demand_shock_cat.retail_ds.gold_demand_features
WHERE rolling_std_7 IS NOT NULL
ORDER BY abs_z_score DESC
LIMIT 50;

-- ─────────────────────────────────────────────
-- Query 3: Average demand by day of week
-- ─────────────────────────────────────────────
SELECT
  dow,
  CASE dow
    WHEN 1 THEN 'Sunday'
    WHEN 2 THEN 'Monday'
    WHEN 3 THEN 'Tuesday'
    WHEN 4 THEN 'Wednesday'
    WHEN 5 THEN 'Thursday'
    WHEN 6 THEN 'Friday'
    WHEN 7 THEN 'Saturday'
  END AS day_name,
  AVG(demand)  AS avg_demand,
  COUNT(*)     AS row_count
FROM demand_shock_cat.retail_ds.gold_demand_features
GROUP BY dow
ORDER BY dow;

-- ─────────────────────────────────────────────
-- Query 4: Store / item seasonality sketch (avg demand per month)
-- ─────────────────────────────────────────────
SELECT
  store_id,
  item_id,
  month,
  AVG(demand) AS avg_demand_by_month
FROM demand_shock_cat.retail_ds.gold_demand_features
GROUP BY store_id, item_id, month
ORDER BY store_id, item_id, month;

-- ─────────────────────────────────────────────
-- Query 5: Most volatile items (highest std dev of demand)
-- ─────────────────────────────────────────────
SELECT
  store_id,
  item_id,
  STDDEV(demand)  AS demand_std,
  AVG(demand)     AS demand_avg,
  COUNT(*)        AS n_days
FROM demand_shock_cat.retail_ds.gold_demand_features
GROUP BY store_id, item_id
ORDER BY demand_std DESC
LIMIT 20;

-- ─────────────────────────────────────────────
-- Query 6: Join predictions back and compute hit rate
--           (fraction of predicted shocks that were true shocks)
-- ─────────────────────────────────────────────
SELECT
  p.model_name,
  p.model_version,
  COUNT(*)                                                        AS total_scored,
  SUM(p.is_shock_pred)                                            AS predicted_shocks,
  SUM(g.is_shock_label)                                           AS actual_shocks,
  SUM(CASE WHEN p.is_shock_pred = 1 AND g.is_shock_label = 1
           THEN 1 ELSE 0 END)                                    AS true_positives,
  SUM(CASE WHEN p.is_shock_pred = 1 AND g.is_shock_label = 1
           THEN 1 ELSE 0 END)
    / NULLIF(SUM(p.is_shock_pred), 0)                            AS precision_shock,
  SUM(CASE WHEN p.is_shock_pred = 1 AND g.is_shock_label = 1
           THEN 1 ELSE 0 END)
    / NULLIF(SUM(g.is_shock_label), 0)                           AS recall_shock
FROM demand_shock_cat.retail_ds.gold_predictions_daily p
LEFT JOIN demand_shock_cat.retail_ds.gold_demand_features g
  ON  p.date     = g.date
  AND p.store_id = g.store_id
  AND p.item_id  = g.item_id
GROUP BY p.model_name, p.model_version
ORDER BY p.model_version DESC;
