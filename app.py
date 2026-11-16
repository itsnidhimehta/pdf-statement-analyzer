"""Step 3 - Present: a Streamlit web app on top of the extractor and analyser.

Run with:  streamlit run app.py
"""

import logging

import streamlit as st

from analyzer import clean_transactions, monthly_summary, overall_summary, spend_by_category
from extractor import extract_transactions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("app")

st.set_page_config(page_title="PDF Statement Analyser", layout="wide")
st.title("PDF Statement Analyser")
st.write("Upload a bank statement PDF to see where the money came from and where it went.")

uploaded_file = st.file_uploader("Upload statement (PDF)", type="pdf")

if uploaded_file is None:
    st.info("Waiting for a PDF. You can try the app with sample_statement.pdf.")
    st.stop()

try:
    raw = extract_transactions(uploaded_file)
    df = clean_transactions(raw)
except ValueError as error:
    # Expected problems (no table, missing column) - show a clear message.
    logger.error("Could not process %s: %s", uploaded_file.name, error)
    st.error(str(error))
    st.stop()
except Exception:
    # Anything unexpected - log full details, show a simple message to the user.
    logger.exception("Unexpected failure while processing %s", uploaded_file.name)
    st.error("Something went wrong while reading this PDF. Please check the file and try again.")
    st.stop()

logger.info("Processed %s: %d clean transaction(s)", uploaded_file.name, len(df))

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
