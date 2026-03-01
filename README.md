# Demand Shock Detector — Lakehouse

## What it does

This project detects unexpected **demand shocks** (sudden spikes or drops) in retail item sales using a full **Azure Databricks + Unity Catalog + Delta Lake** Medallion architecture. Raw M5 Forecasting data flows through bronze → silver → gold Delta layers, where rich lag and rolling-window features are engineered. A **RandomForest** model is trained, logged to **MLflow**, and registered in the Unity Catalog Model Registry. A **FastAPI** microservice exposes real-time predictions and shock scores via REST API, while scheduled Databricks Jobs handle nightly batch inference that writes back to the `gold_predictions_daily` table.

---

## Architecture

```
CSV (DBFS / data/sample/)
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  BRONZE   demand_shock_cat.retail_ds.bronze_demand_raw          │
│           • raw strings, ingested_at, source_file               │
└──────────────────────┬──────────────────────────────────────────┘
                       │  clean / cast / dedupe
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  SILVER   demand_shock_cat.retail_ds.silver_demand_clean        │
│           • typed (date DATE, demand DOUBLE), row_hash          │
└──────────────────────┬──────────────────────────────────────────┘
                       │  feature engineering
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  GOLD     demand_shock_cat.retail_ds.gold_demand_features       │
│           • lag_1, lag_7, rolling_mean_7, rolling_std_7         │
│           • dow, weekofyear, month                              │
│           • target_next_day, is_shock_label                     │
└──────────────────────┬──────────────────────────────────────────┘
          ┌────────────┴───────────────────────────┐
          ▼ train                                  ▼ batch score
  MLflow Experiment                    gold_predictions_daily
  → Model Registry                     (y_hat, shock_score, is_shock_pred)
  → "demand_shock_detector" Champion
          │
          ▼
  FastAPI /predict (real-time)
```

Architecture diagram: [`docs/architecture.png`](docs/architecture.png)

---

## Unity Catalog Usage

| Object    | Name                                              |
|-----------|---------------------------------------------------|
| Catalog   | `demand_shock_cat`                                |
| Schema    | `demand_shock_cat.retail_ds`                      |
| Table     | `demand_shock_cat.retail_ds.bronze_demand_raw`    |
| Table     | `demand_shock_cat.retail_ds.silver_demand_clean`  |
| Table     | `demand_shock_cat.retail_ds.gold_demand_features` |
| Table     | `demand_shock_cat.retail_ds.gold_predictions_daily` |
| Model     | `demand_shock_detector` (alias: `Champion`)       |

Screenshots (add after running in your workspace):
- [`docs/unity_catalog_tree.png`](docs/unity_catalog_tree.png)
- [`docs/unity_catalog_lineage.png`](docs/unity_catalog_lineage.png)

---

## Lakehouse Layers

| Layer  | Table                    | Contents                                                     |
|--------|--------------------------|--------------------------------------------------------------|
| Bronze | `bronze_demand_raw`      | Raw string-typed CSV rows + `ingested_at` + `source_file`   |
| Silver | `silver_demand_clean`    | Typed, filtered (demand ≥ 0), deduplicated, `row_hash`      |
| Gold   | `gold_demand_features`   | Lag/rolling features, calendar features, shock label        |
| Gold   | `gold_predictions_daily` | Model predictions, shock scores, metadata                   |

---

## Modeling + MLflow

Model: `sklearn.ensemble.RandomForestRegressor` (n_estimators=100, max_depth=10)

**Feature columns:** `lag_1`, `lag_7`, `rolling_mean_7`, `rolling_std_7`, `dow`, `weekofyear`, `month`

**Label:** `target_next_day` (next-day demand)

**Shock definition:** `abs_z = abs(demand - rolling_mean_7) / (rolling_std_7 + 1e-6) ≥ 3.0`

### Example run metrics

| Metric | Value       |
|--------|-------------|
| MAE    | ~2.4        |
| RMSE   | ~4.1        |

*(Run your own training to get exact numbers from your dataset.)*

MLflow artifacts logged per run:
- Pickled model
- `reports/feature_importance.csv`
- `reports/training_summary.json`

---

## API Usage

### Start the service locally

```bash
cd service
pip install -r requirements.txt
MLFLOW_TRACKING_URI=http://<your-databricks-host>/mlflow \
  MODEL_NAME=demand_shock_detector \
  uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Health check

```bash
curl http://localhost:8000/health
# {"status":"ok","timestamp":"2024-01-15T08:00:00+00:00"}
```

### Model info

```bash
curl http://localhost:8000/model-info
# {"model_name":"demand_shock_detector","model_version":"3","loaded_at":"...","mlflow_tracking_uri":"..."}
```

### Predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "lag_1": 12.5,
    "lag_7": 11.0,
    "rolling_mean_7": 12.0,
    "rolling_std_7": 1.5,
    "dow": 3,
    "weekofyear": 20,
    "month": 5
  }'
# {
#   "y_hat": 12.8,
#   "shock_score": 0.53,
#   "is_shock_pred": 0,
#   "model_name": "demand_shock_detector",
#   "model_version": "3"
# }
```

---

## How to Run

### Databricks Steps

1. **Upload sample data to DBFS:**
   ```
   dbutils.fs.cp("file:/path/to/data/sample/demand_sample.csv",
                 "dbfs:/FileStore/demand_shock/sample/demand_sample.csv")
   ```

2. **Run SQL setup** (in Databricks SQL Warehouse):
   ```sql
   -- Run in order:
   -- databricks/sql/00_create_catalog_schema.sql
   -- databricks/sql/01_create_bronze_tables.sql
   -- databricks/sql/02_create_silver_tables.sql
   -- databricks/sql/03_create_gold_tables.sql
   -- databricks/sql/04_grants.sql
   ```

3. **Run notebooks in order** (attach to a cluster with Unity Catalog):
   - `databricks/notebooks/00_config.py`
   - `databricks/notebooks/01_bronze_ingest.py`
   - `databricks/notebooks/02_silver_clean.py`
   - `databricks/notebooks/03_gold_features.py`
   - `databricks/notebooks/04_train_model_mlflow.py`
   - `databricks/notebooks/06_register_model.py`
   - `databricks/notebooks/05_batch_scoring_writeback.py`

4. **Schedule jobs** using the JSON specs in `databricks/jobs/`:
   - Import `job_bronze_to_gold.json` for the nightly ETL pipeline
   - Import `job_train_and_register.json` for weekly training
   - Import `job_batch_score.json` for daily scoring

### Service Steps

```bash
# Local dev
cd service
pip install -r requirements.txt
export MLFLOW_TRACKING_URI=http://<databricks-workspace>/mlflow
export MODEL_NAME=demand_shock_detector
uvicorn app.main:app --reload

# Docker
docker compose up --build
```

### Generate sample data locally

```bash
pip install pandas numpy
python scripts/make_sample_small.py
# → data/sample/demand_sample.csv (30,000 rows)
```

### Run tests

```bash
pip install -r requirements-dev.txt
pytest service/tests/ -v
```

---

## Resume Bullets

- **Designed and implemented** a production-grade Demand Shock Detector on **Azure Databricks** using a **Medallion lakehouse architecture** (bronze / silver / gold Delta Lake tables) governed by **Unity Catalog**.
- **Engineered ML features** at scale with PySpark window functions (lag-1, lag-7, 7-day rolling mean/std, calendar features) and defined a data-driven shock label (`|z| ≥ 3σ`).
- **Trained and tracked** a RandomForest demand forecasting model with **MLflow** (params, metrics, feature importance, training summary) and registered the Champion version in the Unity Catalog Model Registry.
- **Built and deployed** a **FastAPI** inference microservice (Docker/docker-compose) that loads the Champion model on startup and returns real-time demand predictions with shock scores.
- **Implemented scheduled Databricks Jobs** (Workflow JSON) for nightly bronze→gold ETL, weekly model retraining, and daily batch scoring with write-back to `gold_predictions_daily`.
- **Wrote end-to-end tests** (FastAPI TestClient + monkeypatching) achieving 100% coverage of API endpoints and shock score logic.