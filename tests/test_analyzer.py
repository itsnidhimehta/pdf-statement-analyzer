"""Unit tests for the cleaning and analysis logic.  Run with:  pytest"""

import pandas as pd
import pytest

from analyzer import (
    categorise,
    clean_transactions,
    find_balance_mismatches,
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


def test_categorise_matches_whole_words_only():
    assert categorise("LIC PREMIUM") == "Other"  # "premium" contains "emi" - must not be Loan / EMI
    assert categorise("HOME LOAN EMI") == "Loan / EMI"


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


def test_bank_column_names_and_excel_dates_are_understood():
    raw = pd.DataFrame(
        {
            "Txn Date": ["2026-09-01 00:00:00", "02/09/2026"],
            "Narration": ["SALARY CREDIT", "RENT NEFT"],
            "Withdrawal Amt.": ["", "18000"],
            "Deposit Amt.": ["65000", ""],
            "Closing Balance (INR)": ["90000", "72000"],
        }
    )
    df = clean_transactions(raw)
    assert list(df["date"].dt.strftime("%d/%m/%Y")) == ["01/09/2026", "02/09/2026"]
    assert list(df["debit"]) == [0.0, 18000.0]
    assert list(df["balance"]) == [90000.0, 72000.0]


def test_balance_check_flags_a_misread_amount():
    raw = pd.DataFrame(
        {
            "Date": ["01/09/2026", "02/09/2026", "03/09/2026"],
            "Description": ["SALARY", "RENT", "SWIGGY"],
            "Debit": ["", "18,000.00", "4,500.00"],  # really 450.00 - an OCR slip
            "Credit": ["65,000.00", "", ""],
            "Balance": ["90,000.00", "72,000.00", "71,550.00"],
        }
    )
    mismatches = find_balance_mismatches(clean_transactions(raw))
    assert list(mismatches["description"]) == ["SWIGGY"]
    assert mismatches["expected_balance"].iloc[0] == 67500.00


def test_summaries():
    df = clean_transactions(make_raw())
    summary = overall_summary(df)
    assert summary["total_credit"] == 65000.00
    assert summary["total_debit"] == 2949.00
    assert summary["net_flow"] == 62051.00
    assert spend_by_category(df).index[0] == "Shopping"
    assert list(monthly_summary(df).index) == ["2026-09", "2026-10"]
