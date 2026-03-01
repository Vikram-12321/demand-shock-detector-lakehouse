"""transform.py — Feature transformation helpers for the API."""


def compute_shock_score(
    rolling_mean_7: float,
    rolling_std_7: float,
    y_hat: float,
    actual: float | None = None,
) -> float:
    """Compute a shock score for the predicted demand.

    Formula (API version — no actual demand required):
        shock_score = abs(y_hat - rolling_mean_7) / (rolling_std_7 + 1e-6)

    If `actual` is provided (e.g. in offline evaluation), you can substitute
    y_hat with actual to measure the realised shock:
        shock_score = abs(actual - rolling_mean_7) / (rolling_std_7 + 1e-6)

    For the real-time API we always use y_hat because no actual is available.

    Args:
        rolling_mean_7: 7-day rolling mean of demand (from request).
        rolling_std_7:  7-day rolling std dev of demand (from request).
        y_hat:          Model-predicted next-day demand.
        actual:         Observed demand (optional; not used in API path).

    Returns:
        Shock score (non-negative float).
    """
    return abs(y_hat - rolling_mean_7) / (rolling_std_7 + 1e-6)
