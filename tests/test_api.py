"""Tests for the FastAPI endpoints.  Run with:  pytest"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import api

ROOT = Path(__file__).parent.parent
client = TestClient(api.app)


def upload(name: str, folder: Path = ROOT / "samples"):
    path = folder / name
    return {"file": (path.name, path.read_bytes())}


def test_health(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert client.get("/health").json() == {"status": "ok", "ai_configured": False}


def test_analyze_csv_returns_summary_and_balance_check():
    response = client.post("/analyze", files=upload("statement.csv"))
    assert response.status_code == 200
    body = response.json()
    assert body["read_method"] == "CSV"
    assert body["summary"]["transactions"] == 43
    assert body["balance_check"]["passed"] is True
    assert body["transactions"][0]["date"] == "01/09/2026"


def test_analyze_rejects_unsupported_file():
    response = client.post("/analyze", files={"file": ("notes.txt", b"hello")})
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_analyze_reports_file_without_a_table():
    response = client.post("/analyze", files={"file": ("empty.csv", b"just,some\nrandom,text\n")})
    assert response.status_code == 422


def test_ask_without_key_gives_clear_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    response = client.post("/ask", files=upload("statement.csv"), data={"question": "Total food spend?"})
    assert response.status_code == 503


def test_ask_with_ai(monkeypatch):
    class FakeLLM:
        replies = [
            AIMessage(content="", tool_calls=[{
                "name": "total_spent_or_received", "id": "c1", "args": {"flow": "debit", "category": "Rent"},
            }]),
            AIMessage(content="You paid ₹54,000.00 in rent."),
        ]

        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            return self.replies.pop(0)

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(api, "make_llm", lambda key: FakeLLM())
    response = client.post("/ask", files=upload("statement.csv"), data={"question": "Total rent?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "You paid ₹54,000.00 in rent."
    assert body["steps"][0]["result"]["total"] == 54000.00  # 3 months x ₹18,000, computed by pandas
