"""Step 1 - Extract: read the transaction table out of any supported statement file.

Supported formats and how each one is read:

    Digital PDF        -> pdfplumber reads the table directly
    Scanned PDF        -> each page is turned into an image, then OCR (Tesseract)
    Image (PNG/JPG)    -> OCR (Tesseract)
    Excel (XLSX/XLS)   -> pandas.read_excel
    CSV                -> pandas.read_csv
    Word (DOCX)        -> python-docx reads the tables

Every reader returns the same thing - a raw DataFrame of text - so the cleaning
and analysis code in analyzer.py does not care which format the file came in.
"""

import logging
from pathlib import Path

import pandas as pd
import pdfplumber
from docx import Document
from PIL import Image

from ocr import ocr_images

logger = logging.getLogger(__name__)

SUPPORTED_TYPES = ["pdf", "png", "jpg", "jpeg", "tif", "tiff", "xlsx", "xls", "csv", "docx"]

OCR_RESOLUTION = 300  # DPI used when turning scanned PDF pages into images
MIN_TEXT_CHARS = 20   # a page with less real text than this is treated as scanned


def extract_transactions(source, file_name: str) -> tuple[pd.DataFrame, str]:
    """Read a statement file and return (raw DataFrame, how it was read).

    source can be a file path or an uploaded file object.
    file_name is used to work out the format from its extension.
    Raises ValueError for unsupported files or files with no readable table.
    """
    extension = Path(file_name).suffix.lower().lstrip(".")
    logger.info("Reading %s as %s", file_name, extension or "unknown type")

    if extension == "pdf":
        return _read_pdf(source)
    if extension in ("png", "jpg", "jpeg", "tif", "tiff"):
        return _read_image(source), "Image - OCR (Tesseract)"
    if extension in ("xlsx", "xls"):
        return _read_excel(source), "Excel"
    if extension == "csv":
        return _read_csv(source), "CSV"
    if extension == "docx":
        return _read_word(source), "Word document"

    raise ValueError(
        f"'.{extension}' files are not supported. "
        f"Please upload one of: {', '.join(SUPPORTED_TYPES).upper()}."
    )


# ---------- PDF (digital or scanned) ----------

def _read_pdf(source) -> tuple[pd.DataFrame, str]:
    """Try the fast digital route first; fall back to OCR if the PDF is scanned."""
    with pdfplumber.open(source) as pdf:
        logger.info("Opened PDF with %d page(s)", len(pdf.pages))

        tables = [table for page in pdf.pages for table in page.extract_tables()]
        if tables:
            return _tables_to_dataframe(tables), "Digital PDF"

        text = "".join(page.extract_text() or "" for page in pdf.pages)
        if len(text.strip()) >= MIN_TEXT_CHARS:
            raise ValueError(
                "This PDF has text but no transaction table could be found in it."
            )

        # No text layer at all -> the pages are pictures, so this is a scanned PDF.
        logger.info("No text layer found - treating as a scanned PDF and running OCR")
        images = [page.to_image(resolution=OCR_RESOLUTION).original for page in pdf.pages]

    return ocr_images(images), "Scanned PDF - OCR (Tesseract)"


# ---------- Images ----------

def _read_image(source) -> pd.DataFrame:
    with Image.open(source) as image:
        image.load()
        # A multi-page TIFF holds several pages; other images have just one.
        pages = []
        for frame in range(getattr(image, "n_frames", 1)):
            image.seek(frame)
            pages.append(image.convert("RGB"))
    return ocr_images(pages)


# ---------- Excel, CSV ----------

def _read_excel(source) -> pd.DataFrame:
    # header=None: real bank exports often have the bank name and address
    # in the first few rows, so we find the real header row ourselves.
    grid = pd.read_excel(source, header=None, dtype=str)
    return _grid_to_dataframe(grid.fillna("").values.tolist())


def _read_csv(source) -> pd.DataFrame:
    grid = pd.read_csv(source, header=None, dtype=str, skip_blank_lines=True,
                       on_bad_lines="skip", engine="python")
    return _grid_to_dataframe(grid.fillna("").values.tolist())


# ---------- Word ----------

def _read_word(source) -> pd.DataFrame:
    document = Document(source)
    tables = [
        [[cell.text for cell in row.cells] for row in table.rows]
        for table in document.tables
    ]
    if not tables:
        raise ValueError("No table found in this Word document.")
    return _tables_to_dataframe(tables)


# ---------- Shared helpers ----------

def _is_header(cells: list[str]) -> bool:
    """A header row is one that has a 'date' column and at least one other known heading."""
    lowered = [cell.lower() for cell in cells]
    has_date = any("date" in cell for cell in lowered)
    has_other = any(
        word in cell
        for cell in lowered
        for word in ("description", "narration", "particulars", "debit", "credit", "withdrawal", "deposit")
    )
    return has_date and has_other


def _grid_to_dataframe(grid: list[list]) -> pd.DataFrame:
    """Find the header row inside a grid of cells and return everything below it."""
    rows = [[_clean_cell(cell) for cell in row] for row in grid]
    for index, row in enumerate(rows):
        if _is_header(row):
            header = row
            body = [r for r in rows[index + 1:] if any(r) and r != header]
            if not body:
                break
            logger.info("Extracted %d raw row(s)", len(body))
            return pd.DataFrame(body, columns=header)
    raise ValueError("No transaction table with a Date column was found in this file.")


def _tables_to_dataframe(tables: list[list[list]]) -> pd.DataFrame:
    """Join tables that may be split across pages (header repeats on each page)."""
    header = None
    rows = []
    for table in tables:
        for row in table:
            cells = [_clean_cell(cell) for cell in row]
            if header is None:
                if _is_header(cells):
                    header = cells
                continue  # skip anything above the first header
            if cells == header:
                continue  # the header repeats on every page - skip it
            if len(cells) != len(header):
                logger.warning("Skipped row with unexpected column count: %s", cells)
                continue
            rows.append(cells)

    if header is None or not rows:
        raise ValueError("No transaction table with a Date column was found in this file.")

    logger.info("Extracted %d raw row(s)", len(rows))
    return pd.DataFrame(rows, columns=header)


def _clean_cell(cell) -> str:
    """Turn an empty cell (None/NaN) into '' and remove extra spaces."""
    if cell is None:
        return ""
    text = str(cell).strip()
    return "" if text.lower() == "nan" else text
