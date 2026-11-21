"""Creates sample statements in every supported format (all made-up data).

Run with:  python make_samples.py

Output:
    sample_statement.pdf             digital PDF (from make_sample_pdf.py)
    samples/scanned_statement.pdf    scanned PDF - pages are pictures, no text layer
    samples/statement_photo.jpg      phone-style photo of page 1 (slightly tilted)
    samples/statement.xlsx           Excel export with bank-style column names
    samples/statement.csv            CSV export
    samples/statement.docx           Word document with a table
"""

from pathlib import Path

import pandas as pd
import pdfplumber
from docx import Document
from PIL import Image, ImageFilter

import make_sample_pdf

SAMPLES = Path(__file__).parent / "samples"
HEADER = ["Date", "Description", "Debit", "Credit", "Balance"]


def make_scanned_pdf(digital_pdf: str) -> list[Image.Image]:
    """Render each page of the digital PDF to a picture and save the pictures as a PDF.

    This is exactly what a scanner does: the result has no text, only images.
    """
    with pdfplumber.open(digital_pdf) as pdf:
        pages = [page.to_image(resolution=200).original.convert("RGB") for page in pdf.pages]
    # A light blur makes it look like a real scan rather than a perfect render.
    pages = [page.filter(ImageFilter.GaussianBlur(0.6)) for page in pages]
    pages[0].save(SAMPLES / "scanned_statement.pdf", save_all=True, append_images=pages[1:])
    return pages


def make_photo(page: Image.Image) -> None:
    """A phone photo of page 1: slightly rotated and saved as JPEG."""
    photo = page.rotate(0.7, resample=Image.BICUBIC, expand=True, fillcolor="white")
    photo.save(SAMPLES / "statement_photo.jpg", quality=85)


def make_excel(rows: list[list[str]]) -> None:
    """Excel export with a bank title above the table and bank-style column names."""
    excel_header = ["Txn Date", "Narration", "Withdrawal Amt", "Deposit Amt", "Closing Balance"]
    table = pd.DataFrame(rows, columns=excel_header)
    with pd.ExcelWriter(SAMPLES / "statement.xlsx") as writer:
        pd.DataFrame([["Sample Bank - Account Statement (demo data)"], [""]]).to_excel(
            writer, header=False, index=False
        )
        table.to_excel(writer, index=False, startrow=2)


def make_csv(rows: list[list[str]]) -> None:
    pd.DataFrame(rows, columns=HEADER).to_csv(SAMPLES / "statement.csv", index=False)


def make_word(rows: list[list[str]]) -> None:
    document = Document()
    document.add_heading("Sample Bank - Account Statement (demo data)", level=1)
    table = document.add_table(rows=1, cols=len(HEADER))
    table.style = "Table Grid"
    for cell, text in zip(table.rows[0].cells, HEADER):
        cell.text = text
    for row in rows:
        for cell, text in zip(table.add_row().cells, row):
            cell.text = text
    document.save(SAMPLES / "statement.docx")


def main():
    SAMPLES.mkdir(exist_ok=True)
    digital_pdf = str(Path(__file__).parent / "sample_statement.pdf")
    make_sample_pdf.main(digital_pdf)

    rows = make_sample_pdf.build_rows()
    pages = make_scanned_pdf(digital_pdf)
    make_photo(pages[0])
    make_excel(rows)
    make_csv(rows)
    make_word(rows)
    print(f"Created sample files in {SAMPLES}")


if __name__ == "__main__":
    main()
