import base64
import json

import pytest
from fastapi.testclient import TestClient

# 1x1 PNG
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("MOCK_VLM", "1")
    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    from app.main import app
    return TestClient(app)


def test_photo_without_note_asks_for_confirmation(client):
    r = client.post("/api/identify", files={"image": ("p.png", PNG, "image/png")}, data={"region": "IN-KL"})
    assert r.status_code == 200
    body = r.json()
    assert body["guidance"]["stream"] == "dry"
    # mock returns a low-confidence bottle with "battery" as an alternative -> hazard doubt
    assert body["confirm"]["needed"] is True
    assert "battery" in body["confirm"]["reason"]
    assert any(c["item_type"] == "battery" and c["hazardous"] for c in body["confirm"]["candidates"])


def test_text_only_swollen_battery(client):
    r = client.post("/api/identify", data={"note": "swollen phone battery", "region": "IN-KL"})
    body = r.json()
    assert body["identification"]["item_type"] == "battery"
    assert body["guidance"]["rule_id"] == "R002"
    assert body["guidance"]["hazard"] is True


def test_requires_photo_or_text(client):
    assert client.post("/api/identify", data={"note": "  "}).status_code == 400


def test_rejects_non_image(client):
    r = client.post("/api/identify", files={"image": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_correction_and_feedback_are_logged(client, tmp_path):
    r = client.post("/api/guidance", json={"item_type": "cfl_or_tube_light", "condition": "damaged", "region": "IN", "request_id": "abc"})
    assert r.json()["guidance"]["rule_id"] == "R021"
    r = client.post("/api/feedback", json={"request_id": "abc", "helpful": False, "comment": "wrong bin"})
    assert r.json() == {"ok": True}
    lines = (tmp_path / "feedback.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[-1])["comment"] == "wrong bin"
    assert (tmp_path / "corrections.jsonl").exists()


def test_missing_token_gives_clear_error(client, monkeypatch):
    monkeypatch.setenv("MOCK_VLM", "0")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    r = client.post("/api/identify", data={"note": "battery"})
    assert r.status_code == 502
    assert "HF_TOKEN" in r.json()["detail"]


def test_serves_mobile_site(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Tap to take a photo" in r.text
    assert client.get("/app.js").status_code == 200
    assert client.get("/api/config").json()["item_types"]  # API routes win over frontend files


def test_logs_to_stdout_on_vercel(client, monkeypatch, capsys):
    import app.main as main
    monkeypatch.setattr(main, "ON_VERCEL", True)
    monkeypatch.delenv("LOG_DIR", raising=False)
    client.post("/api/feedback", json={"request_id": "v1", "helpful": True})
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert line["log"] == "feedback" and line["request_id"] == "v1"
