"""test_health.py — Tests for the /health and /model-info endpoints."""

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app

client = TestClient(app, raise_server_exceptions=True)


def test_health_status_ok():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "timestamp" in data


def test_health_returns_json():
    response = client.get("/health")
    assert response.headers["content-type"].startswith("application/json")


def test_model_info_returns_expected_keys():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    for key in ("model_name", "model_version", "loaded_at", "mlflow_tracking_uri"):
        assert key in data, f"Missing key: {key}"
