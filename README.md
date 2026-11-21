# PDF Statement Analyser

**Live demo:** https://pdf-statement-analyzer.streamlit.app

Upload a bank statement in almost any format and get a clean transaction table, spending by category and a month-by-month summary - then ask questions about it in plain English.

**Features**
- Reads digital PDFs, scanned PDFs and photos (OCR), Excel, CSV and Word
- Balance check that catches misread numbers
- **Ask your statement** - AI chat using LLM tool calling (Groq), where pandas does the maths
- **AI categorisation** - keyword rules first, AI only for what the rules can't place
- **REST API** (FastAPI) for other programs: `/analyze` and `/ask`

## Supported formats

| Format | How it is read |
|---|---|
| Digital PDF | pdfplumber reads the table directly |
| Scanned PDF | no text layer found, so each page becomes an image, then OCR (Tesseract) |
| Photo / image (JPG, PNG, TIFF) | OCR (Tesseract) |
| Excel (XLSX, XLS) | pandas |
| CSV | pandas |
| Word (DOCX) | python-docx |

The app works out the format from the file extension. For PDFs it tries the fast digital route first and switches to OCR automatically only if the PDF turns out to be scanned.

## How it works

1. **Extract** (`extractor.py`, `ocr.py`) - picks the right reader for the file type. Every reader returns the same raw table, so the rest of the app doesn't care which format the file came in.
2. **Clean and analyse** (`analyzer.py`) - pandas maps bank-specific column names (Narration, Withdrawal Amt, Deposit Amt...) to standard ones, converts text to dates and numbers, removes duplicates and bad rows, tags each transaction with a category and builds the summaries.
3. **Check** (`analyzer.find_balance_mismatches`) - for every row, *previous balance - debit + credit* must equal the balance shown. Rows that don't add up are flagged, which catches OCR misreads.
4. **Present** (`app.py`) - Streamlit shows the format used, the balance check, totals, charts, a filterable table and a CSV download.

### How the OCR step works (`ocr.py`)

1. **Clean the image** - fix phone rotation, greyscale, boost contrast, upscale small images.
2. **Straighten it (deskew)** - try small angles and keep the one where text lines are most horizontal.
3. **Read words with positions** - Tesseract (`--psm 4`, suited to column-aligned text) returns each word with its x/y position and line number.
4. **Rebuild the table** - the header line shows where Debit, Credit and Balance sit across the page. Each amount on a transaction line is placed under the nearest heading, and the remaining words become the description.

Step 4 is needed because OCR gives plain text, not a table: without word positions there is no way to tell whether `450.00` was a debit or a credit.

## AI features (Groq)

### Ask your statement - tool calling (`ai.py`)

LLMs are good at understanding questions but bad at arithmetic, so the model never calculates anything:

1. The question goes to the LLM along with a short description of the data (months, categories) and two **tools** - plain pandas functions:
   - `total_spent_or_received` - totals, optionally filtered by category / month / merchant, or grouped for rankings
   - `find_transactions` - lists matching transactions
2. The LLM replies with a **tool call**, e.g. `total_spent_or_received(flow="debit", category="Food", month="2026-10")`.
3. pandas runs that function and returns the exact number.
4. The result goes back to the LLM, which writes the answer in a sentence.

The app shows every tool call under "How I worked this out", so each answer can be checked. Guardrails: a maximum of 4 tool rounds, a question length limit, and errors from bad tool arguments are sent back to the model instead of crashing the app.

### AI categorisation

1. Keyword rules categorise most transactions instantly and for free (whole-word matching, so "LIC PREMIUM" is not mistaken for an EMI).
2. Only the descriptions the rules left as "Other" go to the LLM, in **one batched call**, using structured output (a Pydantic schema).
3. Every AI answer must be one of a fixed list of categories - anything else becomes "Other".
4. Only description text is sent - never amounts, balances or dates. Results are cached, and a `category_source` column shows whether each category came from Rules or AI.

### Setting the API key

Get a free key at https://console.groq.com, then copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and put the key in it (this file is git-ignored). On Streamlit Community Cloud, add `GROQ_API_KEY` under the app's **Settings > Secrets**. Without a key, everything except the AI features still works.

## REST API (FastAPI)

```
uvicorn api:app --reload
```

Open http://127.0.0.1:8000/docs to try it in the browser.

| Endpoint | What it does |
|---|---|
| `GET /health` | Server status and whether the AI key is set |
| `POST /analyze` | Upload a file (optional `use_ai=true`) - returns read method, summary, category and monthly totals, balance check and all transactions as JSON |
| `POST /ask` | Upload a file plus a `question` - returns the AI answer and the tool calls used |

The API reuses the same extractor, analyzer and AI code as the Streamlit app. It validates the file type and size (10 MB) and returns clear HTTP errors (400 unsupported type, 413 too large, 422 no table found, 503 AI not configured).

## Run it

1. Install Tesseract OCR (needed only for scanned PDFs and images):
   - **Windows:** install from https://github.com/UB-Mannheim/tesseract/wiki (default folder `C:\Program Files\Tesseract-OCR` is detected automatically)
   - **Mac:** `brew install tesseract`
   - **Linux:** `sudo apt install tesseract-ocr`
2. Set up and run the app:

```
python -m venv venv
venv\Scripts\activate          # Windows (use: source venv/bin/activate on Mac/Linux)
pip install -r requirements.txt
python make_samples.py         # creates sample statements in every format (made-up data)
streamlit run app.py
```

Then upload any file from `samples/`, or click one of the "Try a sample statement" buttons (one per format).

On Streamlit Community Cloud, `packages.txt` installs Tesseract automatically.

## Run the tests

```
pytest
```

The tests check that every format gives exactly the same transactions and totals as the digital PDF, that a tilted phone photo is read with every amount correct, and that the balance check catches a misread amount. OCR tests are skipped if Tesseract isn't installed.

The AI tests use a fake LLM that replays scripted replies, so they need no API key or internet. They check that the tools compute exact numbers, that tool results are fed back to the model, that a looping model is stopped, that invalid AI categories are rejected, and that amounts are never sent to the AI. The API tests use FastAPI's TestClient, and the app tests click through the Streamlit app.

## Production practices used

- Separate modules for extraction, OCR, analysis, AI, API and UI
- LLM tool calling so numbers come from code, not the model
- AI only where rules fall short, with validated structured output
- One common output from every reader, so new formats don't touch the analysis code
- Data-quality check (running balance) to catch OCR errors instead of trusting them silently
- Explicit date formats instead of guessing, so 01/09 is never read as 9 January
- Caching so slow OCR runs once per file
- Logging instead of print statements
- Clear error messages for expected problems (unsupported type, no table, missing column, Tesseract not installed)
- Unit tests with pytest
- Dependencies listed in requirements.txt (Python) and packages.txt (system)
