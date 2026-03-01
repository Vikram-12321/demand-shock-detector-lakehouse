# Databricks Setup — Demand Shock Detector

This guide covers the end-to-end Azure Databricks setup for the project.

---

## Prerequisites

- Azure Databricks workspace (Premium tier) with **Unity Catalog** enabled
- A Unity Catalog metastore attached to the workspace
- A cluster running Databricks Runtime 14.x (ML recommended)
- Databricks CLI or Repos access

---

## Step 1 — Upload sample data to DBFS

```bash
# From your local machine with Databricks CLI configured:
databricks fs cp data/sample/demand_sample.csv \
  dbfs:/FileStore/demand_shock/sample/demand_sample.csv

# Or from within a Databricks notebook:
# dbutils.fs.cp("file:/Workspace/...", "dbfs:/FileStore/demand_shock/sample/demand_sample.csv")
```

---

## Step 2 — Create Unity Catalog objects

Run the SQL scripts in a **Databricks SQL Warehouse** (not a cluster notebook):

```sql
-- 1. Create catalog + schema
%run databricks/sql/00_create_catalog_schema.sql

-- 2. Create tables
%run databricks/sql/01_create_bronze_tables.sql
%run databricks/sql/02_create_silver_tables.sql
%run databricks/sql/03_create_gold_tables.sql

-- 3. Apply grants
%run databricks/sql/04_grants.sql
```

### Resulting Unity Catalog objects

```
demand_shock_cat (CATALOG)
└── retail_ds (SCHEMA)
    ├── bronze_demand_raw          (Delta, managed)
    ├── silver_demand_clean        (Delta, managed)
    ├── gold_demand_features       (Delta, managed)
    └── gold_predictions_daily     (Delta, managed)
```

---

## Step 3 — Run notebooks

Attach each notebook to a Unity Catalog-enabled cluster.

| Order | Notebook                          | Role                                       |
|-------|-----------------------------------|--------------------------------------------|
| 0     | `00_config.py`                    | Shared constants                           |
| 1     | `01_bronze_ingest.py`             | CSV → bronze Delta table                  |
| 2     | `02_silver_clean.py`              | Bronze → silver (clean, typed, deduped)   |
| 3     | `03_gold_features.py`             | Silver → gold feature engineering         |
| 4     | `04_train_model_mlflow.py`        | Train RandomForest, log to MLflow          |
| 5     | `06_register_model.py`            | Register model, set Champion alias         |
| 6     | `05_batch_scoring_writeback.py`   | Score data, write to gold_predictions_daily |

---

## Step 4 — Schedule jobs

Import the job JSON definitions via the Databricks UI or CLI:

```bash
databricks jobs create --json @databricks/jobs/job_bronze_to_gold.json
databricks jobs create --json @databricks/jobs/job_train_and_register.json
databricks jobs create --json @databricks/jobs/job_batch_score.json
```

**Replace** `<your-username>` in the JSON files with your Databricks workspace username before importing.

### Job schedules (default — all paused)

| Job                       | Default Schedule | Description                        |
|---------------------------|------------------|------------------------------------|
| `job_bronze_to_gold`      | Daily at 02:00   | Bronze ingest → gold features      |
| `job_train_and_register`  | Daily at 04:00   | Train + register new Champion model |
| `job_batch_score`         | Daily at 06:30   | Score latest data                  |

---

## MLflow Model Registry

After training, the model is registered as:

- **Model name:** `demand_shock_detector`
- **Alias:** `Champion` (Unity Catalog model registry)
- **Fallback:** `Production` stage (classic model registry)

To view the model in Unity Catalog:
`Catalog Explorer → demand_shock_cat → Models → demand_shock_detector`

---

## Unity Catalog Grants

By default, `04_grants.sql` grants read access to `account users`.

To restrict to a specific user or group:
```sql
GRANT USAGE  ON CATALOG demand_shock_cat          TO `user@example.com`;
GRANT USAGE  ON SCHEMA  demand_shock_cat.retail_ds TO `user@example.com`;
GRANT SELECT ON TABLE   demand_shock_cat.retail_ds.gold_demand_features TO `user@example.com`;
```

---

## Analytics Queries

Six sample queries are available in `databricks/sql/05_sample_queries.sql`:

1. Row counts per layer (data quality check)
2. Top shocks by absolute z-score
3. Average demand by day of week
4. Store/item seasonality sketch (avg demand per month)
5. Most volatile items (highest std dev)
6. Prediction vs actual shock hit-rate

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `CATALOG NOT FOUND` | Enable Unity Catalog on your workspace |
| `Permission denied on catalog` | Run `04_grants.sql` as workspace admin |
| `Model not found` | Run `04_train_model_mlflow.py` then `06_register_model.py` |
| `dbutils not found` | Run notebook inside Databricks, not locally |
