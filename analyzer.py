"""Step 2 - Clean and analyse: turn raw text rows into numbers we can trust."""

import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["date", "description", "debit", "credit"]

# Different banks use different column names for the same thing.
# Each alias is renamed to our standard name before cleaning.
COLUMN_ALIASES = {
    "txn date": "date",
    "transaction date": "date",
    "tran date": "date",
    "value date": "date",
    "narration": "description",
    "particulars": "description",
    "details": "description",
    "remarks": "description",
    "withdrawal": "debit",
    "withdrawals": "debit",
    "withdrawal amt": "debit",
    "debit amount": "debit",
    "dr": "debit",
    "deposit": "credit",
    "deposits": "credit",
    "deposit amt": "credit",
    "credit amount": "credit",
    "cr": "credit",
    "closing balance": "balance",
}

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


# Every category a transaction can have. The keyword rules use some of them;
# the AI may also choose the extra ones, but nothing outside this list.
ALL_CATEGORIES = list(CATEGORY_RULES) + [
    "Entertainment",
    "Insurance",
    "Investment",
    "Income",
    "Groceries",
    "Other",
]


def categorise(description: str) -> str:
    """Return the spending category for one transaction description."""
    # Whole-word match only: a plain "in" check would put "LIC PREMIUM" under
    # Loan / EMI (because "premium" contains "emi") and "COCA COLA" under Transport.
    words = set(re.findall(r"[a-z0-9]+", str(description).lower()))
    for category, keywords in CATEGORY_RULES.items():
        if any(keyword in words for keyword in keywords):
            return category
    return "Other"


def standardise_columns(columns) -> list[str]:
    """Lower-case column names, drop units like '(INR)' and map bank aliases to our names.

    Only the first column that maps to a standard name is renamed, so a statement
    with both 'Txn Date' and 'Value Date' keeps 'Txn Date' as the date.
    """
    result = []
    for col in columns:
        name = str(col).strip().lower()
        for noise in ("(inr)", "(₹)", "(rs)", "(rs.)", "."):
            name = name.replace(noise, "")
        name = " ".join(name.split())
        standard = COLUMN_ALIASES.get(name, name)
        result.append(standard if standard not in result else name)
    return result


# Date formats seen in Indian bank statements, tried in this order.
# Listed explicitly (instead of letting pandas guess) because guessing can
# silently swap day and month - 01/09 must always mean 1 September.
DATE_FORMATS = [
    "%d/%m/%Y",           # 05/09/2026 - most PDF statements
    "%d-%m-%Y",           # 05-09-2026
    "%d/%m/%y",           # 05/09/26
    "%d-%b-%Y",           # 05-Sep-2026
    "%d %b %Y",           # 05 Sep 2026
    "%Y-%m-%d",           # 2026-09-05 - Excel / CSV exports
    "%Y-%m-%d %H:%M:%S",  # 2026-09-05 00:00:00 - Excel date cells
]


def parse_dates(values: pd.Series) -> pd.Series:
    """Turn date text into real dates, trying each known format in turn."""
    text = values.astype(str).str.strip()
    dates = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    for date_format in DATE_FORMATS:
        missing = dates.isna()
        if not missing.any():
            break
        dates[missing] = pd.to_datetime(text[missing], format=date_format, errors="coerce")
    return dates


def to_number(values: pd.Series) -> pd.Series:
    """Text like '1,250.00' becomes 1250.0; anything unreadable becomes NaN."""
    text = values.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(text, errors="coerce")


def clean_transactions(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw extracted table.

    - standardises column names
    - converts dates and amounts from text to real dates and numbers
    - fills blank amounts with 0
    - removes rows with an unreadable date and exact duplicate rows
    - adds 'category' and 'month' columns
    """
    df = raw.copy()
    df.columns = standardise_columns(df.columns)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Statement is missing column(s): {', '.join(missing)}")

    df["date"] = parse_dates(df["date"])

    # Amounts arrive as text like "1,250.00" or "" - make them numbers.
    for col in ["debit", "credit"]:
        df[col] = to_number(df[col]).fillna(0.0)
    if "balance" in df.columns:
        df["balance"] = to_number(df["balance"])  # blank stays blank (NaN), not 0

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


def find_balance_mismatches(df: pd.DataFrame) -> pd.DataFrame:
    """Rows where the running balance does not add up.

    For every row: previous balance - debit + credit should equal this balance.
    A mismatch usually means a number was misread (common with blurry scans
    or photos) or a row is missing. Returns an empty DataFrame if there is no
    balance column or everything adds up.
    """
    if "balance" not in df.columns:
        return df.iloc[0:0]
    expected = df["balance"].shift(1) - df["debit"] + df["credit"]
    difference = (df["balance"] - expected).round(2)
    mismatch = difference.notna() & (difference.abs() > 0.01)
    result = df.loc[mismatch, ["date", "description", "debit", "credit", "balance"]].copy()
    result["expected_balance"] = expected[mismatch].round(2)
    return result
