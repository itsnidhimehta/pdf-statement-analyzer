"""Tests for reading every supported file format.  Run with:  pytest

Each sample file holds the same made-up statement, so every format must give
the same cleaned transactions and the same totals as the digital PDF.
Create the samples first with:  python make_samples.py
"""

import shutil
from pathlib import Path

import pandas as pd
import pytest

from analyzer import clean_transactions, find_balance_mismatches, overall_summary
from extractor import extract_transactions
from ocr import find_money_columns, line_to_row

ROOT = Path(__file__).parent.parent
SAMPLES = ROOT / "samples"
needs_tesseract = pytest.mark.skipif(
    shutil.which("tesseract") is None, reason="Tesseract OCR is not installed"
)


def read(path: Path):
    raw, method = extract_transactions(path, path.name)
    return clean_transactions(raw), method


@pytest.fixture(scope="module")
def expected():
    df, _ = read(ROOT / "sample_statement.pdf")
    return df


@pytest.mark.parametrize(
    "file_name, method",
    [
        ("statement.xlsx", "Excel"),
        ("statement.csv", "CSV"),
        ("statement.docx", "Word document"),
        pytest.param("scanned_statement.pdf", "Scanned PDF - OCR (Tesseract)", marks=needs_tesseract),
    ],
)
def test_every_format_matches_the_digital_pdf(file_name, method, expected):
    df, used = read(SAMPLES / file_name)
    assert used == method
    columns = ["date", "description", "debit", "credit"]
    pd.testing.assert_frame_equal(df[columns], expected[columns])
    assert overall_summary(df) == overall_summary(expected)
    assert find_balance_mismatches(df).empty


@needs_tesseract
def test_photo_amounts_are_read_correctly(expected):
    # A tilted phone photo of page 1 only - OCR may misspell a word now and
    # then, but every date and amount must be right and the balances must add up.
    df, method = read(SAMPLES / "statement_photo.jpg")
    assert method == "Image - OCR (Tesseract)"
    columns = ["date", "debit", "credit"]
    pd.testing.assert_frame_equal(df[columns], expected[columns].head(len(df)))
    assert len(df) >= 30
    assert find_balance_mismatches(df).empty


def test_unsupported_format_is_rejected():
    with pytest.raises(ValueError, match="not supported"):
        extract_transactions("notes.txt", "notes.txt")


def test_ocr_line_is_split_into_columns():
    # Words as OCR returns them: text plus horizontal position on the page.
    header = [{"text": t, "x": x} for t, x in
              [("Date", 50), ("Description", 200), ("Debit", 500), ("Credit", 650), ("Balance", 800)]]
    columns = find_money_columns(header)
    assert columns == {"Debit": 500, "Credit": 650, "Balance": 800}

    line = [{"text": t, "x": x} for t, x in
            [("03/09/2026", 50), ("UPI-SWIGGY", 180), ("ORDER", 260), ("450.00", 510), ("71,550.00", 790)]]
    row = line_to_row(line, columns)
    assert row == {"Date": "03/09/2026", "Description": "UPI-SWIGGY ORDER",
                   "Debit": "450.00", "Credit": "", "Balance": "71,550.00"}


def test_ocr_ignores_lines_that_are_not_transactions():
    columns = {"Debit": 500, "Credit": 650}
    assert line_to_row([{"text": "Page", "x": 50}, {"text": "1", "x": 90}], columns) is None
