"""AI features (Groq LLM): "Ask your statement" chat and AI categorisation.

Two ideas keep the AI trustworthy:

1. The LLM never does the maths. LLMs are bad at adding up numbers, so for
   questions the model only decides WHAT to look up (which category, which
   month...) by calling a tool. pandas then computes the exact answer.
   This is called "tool calling" (or "function calling").

2. The LLM only fills gaps. Keyword rules categorise most transactions for
   free; the AI is asked only about the ones the rules could not place, in one
   batch, and its answer must be one of our fixed categories.

Analogy: the LLM is a smart receptionist who understands your question and
fills in the request form; pandas is the accountant who does the sums.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Literal

import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from analyzer import ALL_CATEGORIES

logger = logging.getLogger(__name__)

LLM_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
MAX_TOOL_ROUNDS = 4        # stop the model from looping forever
MAX_QUESTION_CHARS = 300   # guardrail on input size
MAX_AI_DESCRIPTIONS = 50   # cap how many descriptions go to the LLM in one call


def make_llm(api_key: str, model: str = LLM_MODEL) -> BaseChatModel:
    """Create the Groq chat model (same setup as my Enterprise RAG project)."""
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=model,
        api_key=api_key,
        temperature=0,              # same question -> same answer
        reasoning_format="hidden",  # gpt-oss "thinks" first; return only the final answer
        max_retries=2,
        timeout=60,
    )


# ----------------------------------------------------------------------------
# Part 1 - tools: plain pandas functions the LLM is allowed to call
# ----------------------------------------------------------------------------

def filter_transactions(
    df: pd.DataFrame,
    category: str | None = None,
    month: str | None = None,
    description_contains: str | None = None,
    min_amount: float | None = None,
) -> pd.DataFrame:
    """Apply the filters the LLM asked for. Every filter is optional."""
    result = df
    if category:
        result = result[result["category"].str.lower() == category.strip().lower()]
    if month:
        result = result[result["month"] == month.strip()]
    if description_contains:
        result = result[
            result["description"].str.contains(description_contains.strip(), case=False, regex=False)
        ]
    if min_amount is not None:
        result = result[(result["debit"] >= min_amount) | (result["credit"] >= min_amount)]
    return result


def summarise(
    df: pd.DataFrame,
    flow: str = "debit",
    group_by: str | None = None,
    top_n: int = 10,
    **filters,
) -> dict:
    """Total money out (debit) or in (credit) after filtering, optionally grouped."""
    rows = filter_transactions(df, **filters)
    column = "credit" if flow == "credit" else "debit"
    rows = rows[rows[column] > 0]
    result = {"filters": {k: v for k, v in filters.items() if v not in (None, "")}, "flow": column}

    if group_by in ("category", "month", "description"):
        groups = rows.groupby(group_by)[column].sum().sort_values(ascending=False).head(top_n)
        result["totals_by_" + group_by] = {key: round(float(value), 2) for key, value in groups.items()}
    else:
        result["transactions"] = int(len(rows))
        result["total"] = round(float(rows[column].sum()), 2)
    return result


def list_rows(df: pd.DataFrame, sort_by: str = "date", limit: int = 15, **filters) -> dict:
    """Return matching transactions (most relevant first), capped at `limit` rows."""
    rows = filter_transactions(df, **filters)
    if sort_by == "amount":
        rows = rows.assign(_amount=rows[["debit", "credit"]].max(axis=1))
        rows = rows.sort_values("_amount", ascending=False)
    else:
        rows = rows.sort_values("date")
    shown = rows.head(limit)
    return {
        "matching_transactions": int(len(rows)),
        "shown": [
            {
                "date": row.date.strftime("%d/%m/%Y"),
                "description": row.description,
                "debit": float(row.debit),
                "credit": float(row.credit),
                "category": row.category,
            }
            for row in shown.itertuples()
        ],
    }


def make_tools(df: pd.DataFrame) -> list:
    """Wrap the pandas functions as LangChain tools bound to this statement."""

    @tool
    def total_spent_or_received(
        flow: Literal["debit", "credit"] = "debit",
        category: str | None = None,
        month: str | None = None,
        description_contains: str | None = None,
        group_by: Literal["category", "month", "description"] | None = None,
        top_n: int = 10,
    ) -> str:
        """Add up money spent (flow='debit') or received (flow='credit').

        Use for any question about totals, sums, 'how much', or rankings.
        Optional filters: category (e.g. 'Food'), month as 'YYYY-MM' (e.g. '2026-10'),
        description_contains (merchant or text, e.g. 'swiggy').
        group_by gives totals per category, month or description (for 'top', 'which', 'compare').
        """
        result = summarise(
            df, flow=flow, group_by=group_by, top_n=top_n,
            category=category, month=month, description_contains=description_contains,
        )
        return json.dumps(result)

    @tool
    def find_transactions(
        category: str | None = None,
        month: str | None = None,
        description_contains: str | None = None,
        min_amount: float | None = None,
        sort_by: Literal["date", "amount"] = "date",
        limit: int = 15,
    ) -> str:
        """List individual transactions that match the filters.

        Use for 'show me', 'list', 'when did I', 'biggest', or transactions above an amount.
        month is 'YYYY-MM'. sort_by='amount' puts the largest first.
        """
        result = list_rows(
            df, sort_by=sort_by, limit=min(limit, 30),
            category=category, month=month,
            description_contains=description_contains, min_amount=min_amount,
        )
        return json.dumps(result)

    return [total_spent_or_received, find_transactions]


# ----------------------------------------------------------------------------
# Part 2 - "Ask your statement": the tool-calling loop
# ----------------------------------------------------------------------------

def build_system_prompt(df: pd.DataFrame) -> str:
    """Tell the model what data exists, so it can choose sensible filters."""
    return f"""You answer questions about ONE bank statement, using the tools provided.

The statement has {len(df)} transactions from {df['date'].min():%d %b %Y} to {df['date'].max():%d %b %Y}.
Months available (YYYY-MM): {', '.join(sorted(df['month'].unique()))}
Categories: {', '.join(sorted(df['category'].unique()))}
Debit = money out (spent). Credit = money in (received). Currency is Indian Rupees (₹).

Rules:
1. ALWAYS call a tool to get numbers. Never calculate, estimate or invent amounts yourself.
2. If you need two numbers (e.g. to compare months), call the tool for each and state both.
3. Use only the category and month values listed above.
4. If the question is not about this statement, say you can only answer questions about it.
5. Answer in 1-3 short sentences. Format money like ₹12,500.00.
"""


def ask_statement(llm: BaseChatModel, df: pd.DataFrame, question: str,
                  history: list[tuple[str, str]] | None = None) -> tuple[str, list[dict]]:
    """Answer a question about the statement.

    Returns (answer, steps) - steps lists every tool call the model made, so the
    app can show HOW the answer was worked out.
    """
    question = question.strip()[:MAX_QUESTION_CHARS]
    tools = make_tools(df)
    tools_by_name = {t.name: t for t in tools}
    model = llm.bind_tools(tools)

    messages = [SystemMessage(content=build_system_prompt(df))]
    for past_question, past_answer in (history or [])[-3:]:  # last few turns give context
        messages += [HumanMessage(content=past_question), AIMessage(content=past_answer)]
    messages.append(HumanMessage(content=question))

    steps = []
    for _ in range(MAX_TOOL_ROUNDS):
        reply = model.invoke(messages)
        messages.append(reply)
        if not reply.tool_calls:
            return str(reply.content).strip(), steps

        for call in reply.tool_calls:
            chosen = tools_by_name.get(call["name"])
            try:
                output = chosen.invoke(call["args"]) if chosen else '{"error": "unknown tool"}'
            except Exception as error:  # bad arguments from the model - tell it, let it retry
                output = json.dumps({"error": str(error)})
            logger.info("Tool %s(%s) -> %s", call["name"], call["args"], output[:200])
            steps.append({"tool": call["name"], "arguments": call["args"], "result": json.loads(output)})
            messages.append(ToolMessage(content=output, tool_call_id=call["id"]))

    return "Sorry, I couldn't work that out. Please try asking in a simpler way.", steps


# ----------------------------------------------------------------------------
# Part 3 - AI categorisation for transactions the keyword rules couldn't place
# ----------------------------------------------------------------------------

class CategorisedTransaction(BaseModel):
    description: str = Field(description="The transaction description, copied exactly")
    category: str = Field(description="Exactly one of the allowed categories")


class CategorisedBatch(BaseModel):
    items: list[CategorisedTransaction]


def ai_categorise(llm: BaseChatModel, descriptions: list[str]) -> dict[str, str]:
    """Ask the LLM to categorise descriptions in ONE call. Returns {description: category}.

    Only the description text is sent - never amounts, balances or dates.
    Any answer that is not one of ALL_CATEGORIES becomes 'Other'.
    """
    descriptions = list(dict.fromkeys(descriptions))[:MAX_AI_DESCRIPTIONS]  # unique, capped
    if not descriptions:
        return {}

    prompt = (
        "Categorise each Indian bank transaction description into exactly one of these "
        f"categories: {', '.join(ALL_CATEGORIES)}.\n"
        "Use 'Other' only if none fits.\n\n"
        + "\n".join(f"- {text}" for text in descriptions)
    )
    structured = llm.with_structured_output(CategorisedBatch)
    batch = structured.invoke([HumanMessage(content=prompt)])

    asked = set(descriptions)
    result = {}
    for item in batch.items:
        if item.description in asked:  # ignore anything we didn't ask about
            result[item.description] = item.category if item.category in ALL_CATEGORIES else "Other"
    logger.info("AI categorised %d of %d description(s)", len(result), len(descriptions))
    return result


def apply_ai_categories(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Fill in 'Other' categories from the AI mapping and record where each category came from."""
    df = df.copy()
    df["category_source"] = "Rules"
    changed = (df["category"] == "Other") & df["description"].isin(mapping)
    df.loc[changed, "category"] = df.loc[changed, "description"].map(mapping)
    df.loc[changed, "category_source"] = "AI"
    return df
