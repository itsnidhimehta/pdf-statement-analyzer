"""OCR: read a transaction table from a picture (a scanned PDF page or a photo).

How it works, in plain words:
1. Tesseract reads the image and gives back every word with its position (x, y).
2. Words are grouped into lines using Tesseract's own line detection
   (it copes with slightly tilted photos better than grouping by height).
3. The header line ("Date ... Debit Credit Balance") tells us where each
   money column sits across the page.
4. For every line that starts with a date, each amount is put under the
   nearest money heading, and the remaining words become the description.

Step 4 matters because OCR returns plain text, not a table - without the
positions we could not tell whether "450.00" was a debit or a credit.
"""

import logging
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytesseract
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# On Windows, Tesseract is usually installed here and is often not on PATH.
# You can also point to it yourself with the TESSERACT_CMD environment variable.
WINDOWS_TESSERACT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
if os.environ.get("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]
elif shutil.which("tesseract") is None and WINDOWS_TESSERACT.exists():
    pytesseract.pytesseract.tesseract_cmd = str(WINDOWS_TESSERACT)

DATE_PATTERN = re.compile(r"^\d{2}[/-]\d{2}[/-]\d{4}$")
AMOUNT_PATTERN = re.compile(r"^\d{1,3}(,\d{2,3})*\.\d{2}$|^\d+\.\d{2}$")

# Heading words we look for on the header line, and the column name each one becomes.
MONEY_HEADINGS = {
    "debit": "Debit",
    "withdrawal": "Debit",
    "withdrawals": "Debit",
    "credit": "Credit",
    "deposit": "Credit",
    "deposits": "Credit",
    "balance": "Balance",
}

MIN_WIDTH = 2000  # small photos are upscaled - Tesseract reads larger text better


def find_skew_angle(image: Image.Image, max_angle: float = 5.0, step: float = 0.1) -> float:
    """Find how many degrees the page is tilted.

    Idea: when text lines are perfectly horizontal, each row of pixels is either
    full of ink (a text line) or empty (the gap between lines), so the ink count
    per row varies a lot. When the page is tilted, ink smears across rows and
    that variation drops. We try small angles and keep the one with the most variation.
    """
    small = image.copy()
    small.thumbnail((800, 800))
    ink = small.point(lambda p: 255 if p < 128 else 0)  # dark pixels become "ink"
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-max_angle, max_angle + step, step):
        rotated = np.asarray(ink.rotate(float(angle), fillcolor=0), dtype=np.float32)
        score = float(rotated.sum(axis=1).var())
        if score > best_score:
            best_angle, best_score = float(angle), score
    return round(best_angle, 2)


def prepare_image(image: Image.Image) -> Image.Image:
    """Clean up the picture before OCR: fix rotation and tilt, greyscale, contrast, size."""
    image = ImageOps.exif_transpose(image)  # phone photos are often stored sideways
    image = ImageOps.grayscale(image)
    image = ImageOps.autocontrast(image)
    angle = find_skew_angle(image)
    if abs(angle) >= 0.2:
        logger.info("Straightening image by %.2f degrees", angle)
        image = image.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=255)
    if image.width < MIN_WIDTH:
        scale = MIN_WIDTH / image.width
        image = image.resize((MIN_WIDTH, int(image.height * scale)), Image.LANCZOS)
    return image


def read_words(image: Image.Image) -> list[dict]:
    """Run Tesseract and return each recognised word with its position."""
    data = pytesseract.image_to_data(
        image,
        config="--psm 4",  # psm 4 = one column of rows; works best for statement tables
        output_type=pytesseract.Output.DICT,
    )
    words = []
    for i, text in enumerate(data["text"]):
        text = text.strip()
        if not text or float(data["conf"][i]) < 0:
            continue
        text = text.strip("|[]_—")  # table grid lines are often read as | or _
        if not text:
            continue
        words.append(
            {
                "text": text,
                "x": data["left"][i] + data["width"][i] / 2,  # centre of the word
                "line": (data["block_num"][i], data["par_num"][i], data["line_num"][i]),
            }
        )
    return words


def group_into_lines(words: list[dict]) -> list[list[dict]]:
    """Group words by the line Tesseract found them on, each line sorted left to right."""
    lines: dict[tuple, list[dict]] = {}
    for word in words:
        lines.setdefault(word["line"], []).append(word)
    return [sorted(line, key=lambda w: w["x"]) for line in lines.values()]


def find_money_columns(line: list[dict]) -> dict | None:
    """If this line is the table header, return {column name: x position}."""
    texts = [w["text"].lower().strip(".:") for w in line]
    if "date" not in texts:
        return None
    columns = {}
    for word, text in zip(line, texts):
        name = MONEY_HEADINGS.get(text)
        if name and name not in columns:
            columns[name] = word["x"]
    # A header needs at least a debit or credit column to be useful.
    return columns if ("Debit" in columns or "Credit" in columns) else None


def line_to_row(line: list[dict], money_columns: dict) -> dict | None:
    """Turn one OCR line into a transaction row, or None if it is not a transaction."""
    first = line[0]["text"]
    if not DATE_PATTERN.match(first):
        return None  # titles, page numbers, footers - not transactions

    row = {"Date": first.replace("-", "/"), "Description": "", **{name: "" for name in money_columns}}
    description_words = []
    for word in line[1:]:
        if AMOUNT_PATTERN.match(word["text"]):
            nearest = min(money_columns, key=lambda name: abs(money_columns[name] - word["x"]))
            row[nearest] = word["text"]
        else:
            description_words.append(word["text"])
    row["Description"] = " ".join(description_words)
    return row


def ocr_images(images: list[Image.Image]) -> pd.DataFrame:
    """OCR one or more page images and return all transactions as one raw DataFrame."""
    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError:
        raise ValueError(
            "This file needs OCR, but Tesseract is not installed on this computer. "
            "See 'Run it' in the README for how to install it."
        )

    rows = []
    money_columns = None  # found on page 1, reused if later pages have no header

    for page_number, image in enumerate(images, start=1):
        lines = group_into_lines(read_words(prepare_image(image)))
        logger.info("OCR page %d: %d line(s) of text", page_number, len(lines))

        for line in lines:
            header = find_money_columns(line)
            if header:
                money_columns = header
                continue
            if money_columns is None:
                continue  # still above the table (bank name, address, etc.)
            row = line_to_row(line, money_columns)
            if row:
                rows.append(row)

    if money_columns is None or not rows:
        raise ValueError(
            "Could not find a transaction table in this image. Make sure the photo or scan "
            "is clear, straight, and shows the Date / Debit / Credit header."
        )

    logger.info("OCR extracted %d raw row(s)", len(rows))
    columns = ["Date", "Description"] + list(money_columns)
    return pd.DataFrame(rows, columns=columns)
