"""Step 3 - Present: a Streamlit web app on top of the extractor and analyser.

Run with:  streamlit run app.py
"""

import hashlib
import io
import logging
import os
from pathlib import Path

import streamlit as st

from analyzer import (
    clean_transactions,
    find_balance_mismatches,
    monthly_summary,
    overall_summary,
    spend_by_category,
)
from extractor import SUPPORTED_TYPES, extract_transactions
from ai import ai_categorise, apply_ai_categories, ask_statement, make_llm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("app")

st.set_page_config(page_title="PDF Statement Analyser", layout="wide")
st.title("PDF Statement Analyser")
st.write(
    "Upload a bank statement to see where the money came from and where it went. "
    "Works with digital PDFs, **scanned PDFs and photos (OCR)**, Excel, CSV and Word files."
)

st.caption("Demo only. Please don't upload your real bank statement. Use the sample statements instead.")

HERE = Path(__file__).parent
SAMPLES = {
    "Digital PDF": HERE / "sample_statement.pdf",
    "Scanned PDF (OCR)": HERE / "samples" / "scanned_statement.pdf",
    "Phone photo (OCR)": HERE / "samples" / "statement_photo.jpg",
    "Excel": HERE / "samples" / "statement.xlsx",
    "CSV": HERE / "samples" / "statement.csv",
    "Word": HERE / "samples" / "statement.docx",
}

st.subheader("Try a sample statement")
st.write("One click - each button loads the same made-up statement in a different format.")
for column, label in zip(st.columns(len(SAMPLES)), SAMPLES):
    if column.button(label, width="stretch"):
        st.session_state["sample"] = label

st.subheader("Or upload your own")
uploaded_file = st.file_uploader(
    "Supported: PDF (digital or scanned), JPG, PNG, TIFF, XLSX, XLS, CSV, DOCX",
    type=SUPPORTED_TYPES,
)

if uploaded_file is not None:
    source = uploaded_file
    file_name = uploaded_file.name
elif st.session_state.get("sample"):
    sample_label = st.session_state["sample"]
    source = SAMPLES[sample_label]
    file_name = source.name
    info_col, download_col = st.columns([3, 1])
    info_col.info(f"Showing sample: **{sample_label}** ({file_name})")
    download_col.download_button(
        "Download this sample file",
        data=source.read_bytes(),
        file_name=file_name,
        width="stretch",
    )
else:
    st.info("Click a sample button above, or upload a statement, to see the analysis.")
    st.stop()


@st.cache_data(show_spinner=False)
def process(file_bytes: bytes, file_name: str):
    """Extract and clean once per file - OCR is slow, so cache the result."""
    raw, method = extract_transactions(io.BytesIO(file_bytes), file_name)
    return clean_transactions(raw), method


try:
    file_bytes = source.getvalue() if uploaded_file is not None else source.read_bytes()
    with st.spinner(f"Reading {file_name}... (scanned files and photos take a few seconds)"):
        df, method = process(file_bytes, file_name)
except ValueError as error:
    # Expected problems (unsupported type, no table, missing column) - show a clear message.
    logger.error("Could not process %s: %s", file_name, error)
    st.error(str(error))
    st.stop()
except Exception:
    # Anything unexpected - log full details, show a simple message to the user.
    logger.exception("Unexpected failure while processing %s", file_name)
    st.error("Something went wrong while reading this file. Please check the file and try again.")
    st.stop()

st.success(f"Read **{file_name}** as: **{method}** - {len(df)} transactions found.")

mismatches = find_balance_mismatches(df)
if not mismatches.empty:
    st.warning(
        f"{len(mismatches)} row(s) don't add up: previous balance - debit + credit "
        "doesn't match the balance shown. A number may have been misread "
        "(common with blurry scans or photos) - please check these rows."
    )
    st.dataframe(mismatches, width="stretch")
elif "balance" in df.columns:
    st.caption("Balance check passed: every row's balance adds up.")

logger.info("Processed %s: %d clean transaction(s)", file_name, len(df))


# ---------- AI setup (optional - the app works without a key) ----------

def get_api_key() -> str | None:
    """Read GROQ_API_KEY from .streamlit/secrets.toml (or Streamlit Cloud secrets), else the environment."""
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass  # no secrets file - that's fine
    return os.getenv("GROQ_API_KEY")


@st.cache_data(show_spinner=False)
def cached_ai_categories(descriptions: tuple[str, ...], _api_key: str) -> dict[str, str]:
    """One LLM call per unique set of descriptions (the leading _ keeps the key out of the cache key)."""
    return ai_categorise(make_llm(_api_key), list(descriptions))


api_key = get_api_key()

if api_key:
    use_ai = st.toggle("Use AI to categorise transactions the keyword rules couldn't place", value=True)
    leftovers = tuple(sorted(df.loc[df["category"] == "Other", "description"].unique()))
    if use_ai and leftovers:
        try:
            with st.spinner(f"AI is categorising {len(leftovers)} transaction(s)..."):
                mapping = cached_ai_categories(leftovers, api_key)
            df = apply_ai_categories(df, mapping)
            ai_rows = df[df["category_source"] == "AI"]
            st.caption(
                f"Keyword rules categorised {len(df) - len(ai_rows)} transactions; "
                f"AI categorised the remaining {len(ai_rows)} (see the 'category_source' column). "
                "Only descriptions are sent to the AI - never amounts or balances."
            )
        except Exception:
            logger.exception("AI categorisation failed")
            st.warning("AI categorisation is unavailable right now - showing keyword-rule categories only.")
else:
    st.info(
        "AI features are off: add a GROQ_API_KEY (see README) to turn on AI categorisation "
        "and the 'Ask your statement' chat."
    )

summary = overall_summary(df)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Transactions", summary["transactions"])
col2.metric("Money in", f"₹{summary['total_credit']:,.2f}")
col3.metric("Money out", f"₹{summary['total_debit']:,.2f}")
col4.metric("Net flow", f"₹{summary['net_flow']:,.2f}")

left, right = st.columns(2)
with left:
    st.subheader("Spend by category")
    st.bar_chart(spend_by_category(df))
with right:
    st.subheader("Money in vs money out, by month")
    st.bar_chart(monthly_summary(df))

st.subheader("Transactions")
categories = sorted(df["category"].unique())
chosen = st.multiselect("Filter by category", categories, default=categories)
filtered = df[df["category"].isin(chosen)]
st.dataframe(filtered, width="stretch")

st.download_button(
    "Download cleaned data (CSV)",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="cleaned_transactions.csv",
    mime="text/csv",
)


# ---------- Ask your statement (AI chat) ----------

st.subheader("Ask your statement")
st.write(
    "Ask a question in plain English. The AI decides what to look up; "
    "pandas calculates the exact numbers, so the AI never guesses an amount."
)

if not api_key:
    st.caption("Add a GROQ_API_KEY to use the chat.")
    st.stop()

# Start a fresh conversation whenever a different file is loaded.
file_id = hashlib.md5(file_bytes).hexdigest()
if st.session_state.get("chat_file") != file_id:
    st.session_state["chat_file"] = file_id
    st.session_state["chat"] = []  # list of (question, answer, steps)

EXAMPLES = [
    "How much did I spend on food in October?",
    "What are my top 3 spending categories?",
    "Which month did I spend the most?",
    "Show all transactions above ₹5,000",
]
question = None
for column, example in zip(st.columns(len(EXAMPLES)), EXAMPLES):
    if column.button(example, width="stretch"):
        question = example
typed = st.chat_input("e.g. How much did I pay in rent in total?")
question = typed or question

for past_question, past_answer, past_steps in st.session_state["chat"]:
    with st.chat_message("user"):
        st.write(past_question)
    with st.chat_message("assistant"):
        st.write(past_answer)
        if past_steps:
            with st.expander("How I worked this out"):
                st.json(past_steps)

if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                history = [(q, a) for q, a, _ in st.session_state["chat"]]
                answer, steps = ask_statement(make_llm(api_key), df, question, history)
        except Exception:
            logger.exception("Chat failed for question: %s", question)
            answer, steps = "Sorry, the AI service is unavailable right now. Please try again.", []
        st.write(answer)
        if steps:
            with st.expander("How I worked this out"):
                st.json(steps)
    st.session_state["chat"].append((question, answer, steps))
