from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "clinical-copilot"


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "up"


def test_analyze_no_interaction():
    r = client.post("/analyze", json={"text": "Dolo 650 twice daily for fever"})
    assert r.status_code == 200
    body = r.json()
    assert body["total_medicines"] >= 1
    assert body["total_interactions"] == 0
    assert body["has_safety_concerns"] is False


def test_analyze_severe_interaction():
    r = client.post(
        "/analyze", json={"text": "Aspirin 75mg + Clopidogrel 75mg + Atorvastatin 10mg"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["total_medicines"] >= 2
    assert body["total_interactions"] >= 1
    assert body["has_safety_concerns"] is True


def test_analyze_moderate():
    r = client.post(
        "/analyze", json={"text": "Metformin 500mg twice daily with Glimepiride 2mg"}
    )
    assert r.status_code == 200
    assert r.json()["total_interactions"] == 1


def test_metrics_exposed():
    client.post("/analyze", json={"text": "Aspirin + Warfarin"})
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "http_requests_total" in r.text
    assert "interactions_detected_total" in r.text
