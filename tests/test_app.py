"""Click through the Streamlit app like a user would.  Run with:  pytest"""

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

import ai
from ai import CategorisedBatch, CategorisedTransaction

APP = str(Path(__file__).parent.parent / "app.py")


class FakeLLM:
    """Categorises Netflix, then answers chat with one tool call + a final answer."""

    def __init__(self):
        self.chat_replies = [
            AIMessage(content="", tool_calls=[{
                "name": "total_spent_or_received", "id": "c1",
                "args": {"flow": "debit", "category": "Food", "month": "2026-10"},
            }]),
            AIMessage(content="You spent ₹1,280.00 on food in October 2026."),
        ]
        self.structured = False

    def with_structured_output(self, schema):
        self.structured = True
        return self

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        if self.structured:
            return CategorisedBatch(items=[
                CategorisedTransaction(description="NETFLIX SUBSCRIPTION", category="Entertainment"),
            ])
        return self.chat_replies.pop(0)


def test_app_works_without_an_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    at = AppTest.from_file(APP, default_timeout=60).run()
    next(b for b in at.button if b.label == "CSV").click().run()
    assert not at.exception
    assert "43 transactions" in at.success[0].value
    assert any("AI features are off" in i.value for i in at.info)


def test_app_ai_categorises_and_answers_chat(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(ai, "make_llm", lambda key, model=None: FakeLLM())
    at = AppTest.from_file(APP, default_timeout=60).run()
    next(b for b in at.button if b.label == "CSV").click().run()
    assert any("AI categorised the remaining" in c.value for c in at.caption)

    next(b for b in at.button if b.label.startswith("How much did I spend on food")).click().run()
    assert not at.exception
    messages = [m.markdown[0].value for m in at.chat_message if m.markdown]
    assert "You spent ₹1,280.00 on food in October 2026." in messages
