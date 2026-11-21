"""Tests for the AI features - no API key or internet needed.  Run with:  pytest

A fake LLM replays scripted replies (tool calls, then a final answer), the same
shape Groq returns. This tests OUR code: that tools compute the right numbers,
the loop feeds results back to the model, and bad AI answers are handled safely.
"""

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from ai import (
    CategorisedBatch,
    CategorisedTransaction,
    ai_categorise,
    apply_ai_categories,
    ask_statement,
    summarise,
)
from analyzer import clean_transactions
from extractor import extract_transactions

SAMPLE_PDF = Path(__file__).parent.parent / "sample_statement.pdf"


@pytest.fixture(scope="module")
def df():
    raw, _ = extract_transactions(SAMPLE_PDF, SAMPLE_PDF.name)
    return clean_transactions(raw)


class FakeLLM:
    """Stands in for ChatGroq: returns the scripted replies in order."""

    def __init__(self, replies=None, structured=None):
        self.replies = list(replies or [])
        self.structured = structured
        self.seen = []  # every message list the model was given

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        self.seen.append(list(messages))
        return self.structured if self.structured is not None else self.replies.pop(0)


# ---------- the pandas tools compute exact numbers ----------

def test_summarise_total_for_category_and_month(df):
    result = summarise(df, flow="debit", category="food", month="2026-10")
    # October food: Swiggy 380 + Zomato 540 + Cafe Coffee Day 360
    assert result["total"] == 1280.00
    assert result["transactions"] == 3


def test_summarise_grouped_by_category_is_sorted(df):
    result = summarise(df, flow="debit", group_by="category", top_n=3)
    totals = list(result["totals_by_category"].values())
    assert len(totals) == 3 and totals == sorted(totals, reverse=True)


# ---------- the tool-calling loop ----------

def test_ask_statement_runs_tool_and_returns_answer(df):
    llm = FakeLLM(
        replies=[
            AIMessage(content="", tool_calls=[{
                "name": "total_spent_or_received", "id": "call_1",
                "args": {"flow": "debit", "category": "Food", "month": "2026-10"},
            }]),
            AIMessage(content="You spent ₹1,280.00 on food in October 2026."),
        ]
    )
    answer, steps = ask_statement(llm, df, "How much did I spend on food in October?")

    assert answer == "You spent ₹1,280.00 on food in October 2026."
    assert steps[0]["result"]["total"] == 1280.00
    # The second model call must have received the tool result to word its answer.
    tool_messages = [m for m in llm.seen[1] if isinstance(m, ToolMessage)]
    assert '"total": 1280.0' in tool_messages[0].content


def test_ask_statement_survives_bad_tool_arguments(df):
    llm = FakeLLM(
        replies=[
            AIMessage(content="", tool_calls=[{
                "name": "total_spent_or_received", "id": "call_1", "args": {"flow": "sideways"},
            }]),
            AIMessage(content="Sorry, please rephrase."),
        ]
    )
    answer, steps = ask_statement(llm, df, "??")
    assert "error" in steps[0]["result"]  # error went back to the model, app didn't crash
    assert answer == "Sorry, please rephrase."


def test_ask_statement_stops_a_looping_model(df):
    looping = AIMessage(content="", tool_calls=[{
        "name": "find_transactions", "id": "call_x", "args": {"category": "Food"},
    }])
    llm = FakeLLM(replies=[looping] * 10)
    answer, steps = ask_statement(llm, df, "Food?")
    assert answer.startswith("Sorry")
    assert len(steps) == 4  # MAX_TOOL_ROUNDS


# ---------- AI categorisation ----------

def test_ai_categorise_keeps_only_valid_categories():
    batch = CategorisedBatch(items=[
        CategorisedTransaction(description="NETFLIX SUBSCRIPTION", category="Entertainment"),
        CategorisedTransaction(description="SIP MUTUAL FUND", category="Crypto"),  # not allowed
        CategorisedTransaction(description="SOMETHING WE NEVER ASKED", category="Food"),
    ])
    mapping = ai_categorise(FakeLLM(structured=batch), ["NETFLIX SUBSCRIPTION", "SIP MUTUAL FUND"])
    assert mapping == {"NETFLIX SUBSCRIPTION": "Entertainment", "SIP MUTUAL FUND": "Other"}


def test_ai_categorise_sends_only_descriptions(df):
    llm = FakeLLM(structured=CategorisedBatch(items=[]))
    ai_categorise(llm, ["APOLLO PHARMACY"])
    prompt = llm.seen[0][0].content
    assert "APOLLO PHARMACY" in prompt
    assert "890" not in prompt  # amounts never leave the app


def test_apply_ai_categories_changes_only_other_rows(df):
    mapping = {"NETFLIX SUBSCRIPTION": "Entertainment", "SALARY CREDIT SEP": "Food"}
    result = apply_ai_categories(df, mapping)
    netflix = result[result["description"] == "NETFLIX SUBSCRIPTION"].iloc[0]
    salary = result[result["description"] == "SALARY CREDIT SEP"].iloc[0]
    assert (netflix["category"], netflix["category_source"]) == ("Entertainment", "AI")
    assert (salary["category"], salary["category_source"]) == ("Salary", "Rules")  # rules win
