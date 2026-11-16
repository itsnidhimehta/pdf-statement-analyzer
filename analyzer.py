"""Step 2 - Clean and analyse: turn raw text rows into numbers we can trust."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["date", "description", "debit", "credit"]

# Keyword rules: if the description contains the keyword, it gets that category.
# Order matters - the first match wins, so "Transfer" stays near the end as the
# fallback for plain UPI/NEFT payments.
CATEGORY_RULES = {
    "Salary": ["salary"],
    "Rent": ["rent"],
    "Loan / EMI": ["emi", "loan"],
    "Food": ["swiggy", "zomato", "restaurant", "cafe"],
    "Shopping": ["amazon", "flipkart", "myntra", "dmart"],
    "Transport": ["uber", "ola", "irctc", "petrol", "metro"],
    "Bills": ["electricity", "recharge", "bill", "broadband"],
    "Cash": ["atm"],
    "Transfer": ["upi", "neft", "imps"],
    "Health": ["medical"],
}


def categorise(description: str) -> str:
    """Return the spending category for one transaction description."""
    text = str(description).lower()
    for category, keywords in CATEGORY_RULES.items():
        if any(keyword in text for keyword in keywords):
            return category
    return "Other"


def clean_transactions(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw extracted table.

    - standardises column names
    - converts dates and amounts from text to real dates and numbers
    - fills blank amounts with 0
    - removes rows with an unreadable date and exact duplicate rows
    - adds 'category' and 'month' columns
    """
    df = raw.copy()
    df.columns = [str(col).strip().lower() for col in df.columns]

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Statement is missing column(s): {', '.join(missing)}")

    # Dates arrive as text like "05/09/2026" (day first, Indian format).
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")

    # Amounts arrive as text like "1,250.00" or "" - make them numbers.
    for col in ["debit", "credit"]:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", "", regex=False), errors="coerce"
        ).fillna(0.0)

    before = len(df)
    df = df.dropna(subset=["date"]).drop_duplicates().reset_index(drop=True)
    logger.info("Cleaning removed %d bad or duplicate row(s)", before - len(df))

    df["category"] = df["description"].apply(categorise)
    df["month"] = df["date"].dt.strftime("%Y-%m")
    return df


def overall_summary(df: pd.DataFrame) -> dict:
    """Headline numbers for the whole statement."""
    total_credit = float(df["credit"].sum())
    total_debit = float(df["debit"].sum())
    return {
        "transactions": int(len(df)),
        "total_credit": total_credit,
        "total_debit": total_debit,
        "net_flow": total_credit - total_debit,
    }


def spend_by_category(df: pd.DataFrame) -> pd.Series:
    """Total money spent (debit) per category, biggest first."""
    spend = df.groupby("category")["debit"].sum()
    return spend[spend > 0].sort_values(ascending=False)


def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Money in (credit) and money out (debit) for each month."""
    return df.groupby("month")[["credit", "debit"]].sum()
