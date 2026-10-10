import os

import httpx

API_URL = os.environ.get("TRIPWIRE_API_URL", "http://app:8000")


def test_health():
    resp = httpx.get(f"{API_URL}/health", timeout=10.0)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in ("ok", "degraded")


def test_investigate():
    resp = httpx.post(
        f"{API_URL}/investigate",
        json={"ticker": "AAPL", "lookback_days": 30},
        timeout=600.0,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["verdict"] in ("flagged", "not flagged", "insufficient data")
    assert isinstance(body["citations"], list)
    assert isinstance(body["validation_issues"], list)


if __name__ == "__main__":
    test_health()
    print("test_health: OK")
    test_investigate()
    print("test_investigate: OK")
