import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from amlegal import api

client = TestClient(api.app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["articles"] == 323


def test_search_finds_bigamy_article():
    r = client.post("/search", json={"question": "በጋብቻ ላይ ጋብቻ መፈጸም ይቻላል?", "k": 5})
    assert r.status_code == 200
    body = r.json()
    assert len(body["articles"]) == 5
    assert 11 in [a["article"] for a in body["articles"]]   # Article 11: Bigamy
    assert "not legal advice" in body["disclaimer"]


def test_validation():
    assert client.post("/search", json={"question": "a"}).status_code == 422
    assert client.post("/search", json={"question": "ጋብቻ", "k": 50}).status_code == 422


def test_ask_without_llm_returns_articles_only(monkeypatch):
    monkeypatch.setattr(api, "get_generator", lambda: None)
    r = client.post("/ask", json={"question": "የጋብቻ ዕድሜ ስንት ነው?"})
    assert r.status_code == 200
    assert r.json()["answer"] is None and r.json()["articles"]


def test_ask_with_llm_keeps_only_valid_citations(monkeypatch):
    def fake_llm(messages):
        return "አይቻልም [አንቀጽ 11]። እንዲሁም [አንቀጽ 999] ይመልከቱ።"   # 999 does not exist
    monkeypatch.setattr(api, "get_generator", lambda: fake_llm)
    r = client.post("/ask", json={"question": "በጋብቻ ላይ ጋብቻ መፈጸም ይቻላል?"})
    body = r.json()
    assert body["answer"].startswith("አይቻልም")
    assert body["cited_articles"] == [11]
    assert body["refused"] is False
