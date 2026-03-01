"""health.py — Health-check helpers."""

from datetime import datetime, timezone


def build_health_response() -> dict:
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}
