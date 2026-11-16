"""Step 1 - Extract: read the transaction table out of a PDF bank statement."""

import logging

import pandas as pd
import pdfplumber

logger = logging.getLogger(__name__)


def extract_transactions(pdf_file) -> pd.DataFrame:
    """Read every table in the PDF and return all rows as one raw DataFrame.

    pdf_file can be a file path or an uploaded file object.
    Raises ValueError if the PDF has no readable table (for example a scanned PDF).
    """
    header = None
    rows = []

    with pdfplumber.open(pdf_file) as pdf:
        logger.info("Opened PDF with %d page(s)", len(pdf.pages))

        for page_number, page in enumerate(pdf.pages, start=1):
            for table in page.extract_tables():
                if not table:
                    continue

                # The first row of the first table is the header (column names).
                if header is None:
                    header = [_clean_cell(cell) for cell in table[0]]

                for row in table:
                    cells = [_clean_cell(cell) for cell in row]
                    if cells == header:
                        continue  # the header repeats on every page - skip it
                    if len(cells) != len(header):
                        logger.warning("Skipped odd row on page %d: %s", page_number, cells)
                        continue
                    rows.append(cells)

    if header is None or not rows:
        raise ValueError(
            "No table found in this PDF. It may be a scanned (image) PDF, "
            "which needs OCR before it can be read."
        )

    logger.info("Extracted %d raw row(s)", len(rows))
    return pd.DataFrame(rows, columns=header)


def _clean_cell(cell) -> str:
    """Turn an empty cell (None) into '' and remove extra spaces."""
    return "" if cell is None else str(cell).strip()
