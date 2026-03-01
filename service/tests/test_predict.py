"""test_predict.py — Tests for the /predict endpoint with a monkeypatched model."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app

client = TestClient(app, raise_server_exceptions=True)

SAMPLE_PAYLOAD = {
    "lag_1": 10.0,
    "lag_7": 9.5,
    "rolling_mean_7": 10.2,
    "rolling_std_7": 1.5,
    "dow": 3,
    "weekofyear": 12,
    "month": 3,
}


class _DummyModel:
    """Minimal model stub that always predicts a constant value."""

    def predict(self, X):
        return np.array([42.0] * len(X))


@pytest.fixture(autouse=True)
def patch_model(monkeypatch):
    """Replace the global model with a deterministic stub for all tests."""
    monkeypatch.setattr(main_module, "_model", _DummyModel())
    monkeypatch.setattr(main_module, "_model_version", "test-v1")


def test_predict_returns_200():
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert response.status_code == 200


def test_predict_response_fields():
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    data = response.json()
    for key in ("y_hat", "shock_score", "is_shock_pred", "model_name", "model_version"):
        assert key in data, f"Missing field: {key}"


def test_predict_y_hat_is_constant():
    """The dummy model should return 42.0."""
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert response.json()["y_hat"] == pytest.approx(42.0)


def test_predict_shock_score_formula():
    """shock_score = abs(y_hat - rolling_mean_7) / (rolling_std_7 + 1e-6)."""
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    data = response.json()
    expected = abs(42.0 - SAMPLE_PAYLOAD["rolling_mean_7"]) / (
        SAMPLE_PAYLOAD["rolling_std_7"] + 1e-6
    )
    assert data["shock_score"] == pytest.approx(expected, rel=1e-4)


def test_predict_is_shock_pred_below_threshold():
    """With rolling_mean_7 close to y_hat, shock should NOT be triggered."""
    payload = {**SAMPLE_PAYLOAD, "rolling_mean_7": 42.5, "rolling_std_7": 5.0}
    response = client.post("/predict", json=payload)
    data = response.json()
    # shock_score = abs(42.0 - 42.5) / (5.0 + 1e-6) ≈ 0.1 → not a shock
    assert data["is_shock_pred"] == 0


def test_predict_is_shock_pred_above_threshold():
    """With rolling_mean_7 far from y_hat and small std, shock SHOULD be triggered."""
    payload = {**SAMPLE_PAYLOAD, "rolling_mean_7": 10.0, "rolling_std_7": 0.1}
    response = client.post("/predict", json=payload)
    data = response.json()
    # shock_score = abs(42.0 - 10.0) / (0.1 + 1e-6) ≈ 320 → shock
    assert data["is_shock_pred"] == 1


def test_predict_missing_field_returns_422():
    incomplete = {k: v for k, v in SAMPLE_PAYLOAD.items() if k != "lag_1"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_model_version_in_response():
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert response.json()["model_version"] == "test-v1"
