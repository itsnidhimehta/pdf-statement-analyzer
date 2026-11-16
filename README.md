# PDF Statement Analyser

Upload a bank statement PDF and get a clean transaction table, spending by category and a month-by-month summary.

## How it works

1. **Extract** (`extractor.py`) - pdfplumber reads the transaction table from every page of the PDF.
2. **Clean and analyse** (`analyzer.py`) - pandas converts text to dates and numbers, removes duplicates and bad rows, tags each transaction with a category and builds the summaries.
3. **Present** (`app.py`) - Streamlit shows the totals, charts, a filterable table and a CSV download.

## Run it

```
python -m venv venv
venv\Scripts\activate          # Windows (use: source venv/bin/activate on Mac/Linux)
pip install -r requirements.txt
python make_sample_pdf.py      # creates sample_statement.pdf (made-up data)
streamlit run app.py
```

Then upload `sample_statement.pdf` in the browser page that opens.

## Run the tests

```
pytest
```

## Expected PDF format

A digital (not scanned) PDF with a table that has the columns Date (DD/MM/YYYY), Description, Debit and Credit. Scanned PDFs are images and would need OCR (for example Tesseract) first.

## Production practices used

- Separate modules for extraction, analysis and UI
- Logging instead of print statements
- Clear error messages for expected problems (no table, missing column)
- Unit tests with pytest
- Dependencies listed in requirements.txt
