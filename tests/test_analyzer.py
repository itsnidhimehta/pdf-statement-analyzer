"""Unit tests for the cleaning and analysis logic.  Run with:  pytest"""

import pandas as pd
import pytest

from analyzer import (
    categorise,
    clean_transactions,
    monthly_summary,
    overall_summary,
    spend_by_category,
)


def make_raw():
    return pd.DataFrame(
        {
            "Date": ["01/09/2026", "03/09/2026", "03/09/2026", "05/10/2026", "not a date"],
            "Description": ["SALARY CREDIT", "UPI-SWIGGY ORDER", "UPI-SWIGGY ORDER", "AMAZON PAY", "JUNK"],
            "Debit": ["", "450.00", "450.00", "2,499.00", "10"],
            "Credit": ["65,000.00", "", "", "", ""],
            "Balance": ["90,000.00", "89,550.00", "89,550.00", "87,051.00", ""],
        }
    )


def test_categorise_matches_keywords():
    assert categorise("UPI-SWIGGY ORDER") == "Food"
    assert categorise("Salary Credit Sep") == "Salary"
    assert categorise("UPI-RAHUL TRANSFER") == "Transfer"
    assert categorise("MEDICAL STORE") == "Health"


def test_clean_converts_amounts_to_numbers():
    df = clean_transactions(make_raw())
    assert df.loc[df["description"] == "AMAZON PAY", "debit"].iloc[0] == 2499.00
    assert df.loc[df["description"] == "SALARY CREDIT", "debit"].iloc[0] == 0.0


def test_clean_removes_duplicates_and_bad_dates():
    df = clean_transactions(make_raw())
    assert len(df) == 3  # 5 raw rows - 1 duplicate - 1 bad date


def test_clean_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing column"):
        clean_transactions(pd.DataFrame({"Date": ["01/09/2026"], "Amount": ["10"]}))


def test_summaries():
    df = clean_transactions(make_raw())
    summary = overall_summary(df)
    assert summary["total_credit"] == 65000.00
    assert summary["total_debit"] == 2949.00
    assert summary["net_flow"] == 62051.00
    assert spend_by_category(df).index[0] == "Shopping"
    assert list(monthly_summary(df).index) == ["2026-09", "2026-10"]
