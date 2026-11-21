"""REST API for the statement analyser (FastAPI).

Run with:   uvicorn api:app --reload
Then open:  http://127.0.0.1:8000/docs   (interactive page to try every endpoint)

Endpoints:
    GET  /health    - is the server up, and is the AI configured?
    POST /analyze   - upload a statement, get the analysis back as JSON
    POST /ask       - upload a statement + a question, get an AI answer

The Streamlit app and this API share the same extractor / analyzer / ai code -
the API is just a different "front door" for other programs to use.
"""

import io
import logging
import os
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from ai import ai_categorise, apply_ai_categories, ask_statement, make_llm
from analyzer import (
    clean_transactions,
    find_balance_mismatches,
    monthly_summary,
    overall_summary,
    spend_by_category,
)
from extractor import SUPPORTED_TYPES, extract_transactions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("api")

MAX_FILE_MB = 10

app = FastAPI(
    title="PDF Statement Analyser API",
    description="Upload a bank statement (PDF, scanned PDF, image, Excel, CSV or Word) and get it analysed.",
    version="1.0",
)


# ---------- response shapes (FastAPI validates these and shows them in /docs) ----------

class Summary(BaseModel):
    transactions: int
    total_credit: float
    total_debit: float
    net_flow: float


class BalanceCheck(BaseModel):
    passed: bool
    mismatched_rows: list[dict]


class AnalyzeResponse(BaseModel):
    file_name: str
    read_method: str
    summary: Summary
    spend_by_category: dict[str, float]
    monthly: dict[str, dict[str, float]]
    balance_check: BalanceCheck
    ai_categorised: int
    transactions: list[dict]
    latency_sec: float


class AskResponse(BaseModel):
    question: str
    answer: str
    steps: list[dict]
    latency_sec: float


# ---------- helpers ----------

def get_llm():
    """Create the LLM only when an AI feature is used, so the API runs without a key too."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="AI is not configured: set GROQ_API_KEY.")
    return make_llm(api_key)


async def read_statement(file: UploadFile):
    """Validate the upload, then extract + clean it. Returns (df, read_method)."""
    extension = (file.filename or "").rsplit(".", 1)[-1].lower()
    if extension not in SUPPORTED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '.{extension}'. Use one of: {', '.join(SUPPORTED_TYPES)}",
        )
    content = await file.read()
    if len(content) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File is larger than {MAX_FILE_MB} MB.")
    try:
        raw, method = extract_transactions(io.BytesIO(content), file.filename)
        return clean_transactions(raw), method
    except ValueError as error:  # expected problems: no table, missing column...
        raise HTTPException(status_code=422, detail=str(error))


def to_records(df) -> list[dict]:
    """DataFrame -> list of plain dicts that JSON can carry (dates as DD/MM/YYYY)."""
    out = df.copy()
    out["date"] = out["date"].dt.strftime("%d/%m/%Y")
    return out.astype(object).where(out.notna(), None).to_dict(orient="records")


# ---------- endpoints ----------

@app.get("/health")
def health():
    return {"status": "ok", "ai_configured": bool(os.getenv("GROQ_API_KEY"))}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(..., description="Bank statement file"),
    use_ai: bool = Form(False, description="Use AI to categorise transactions the rules couldn't place"),
):
    start = time.perf_counter()
    df, method = await read_statement(file)

    ai_count = 0
    if use_ai:
        leftovers = df.loc[df["category"] == "Other", "description"].unique().tolist()
        if leftovers:
            df = apply_ai_categories(df, ai_categorise(get_llm(), leftovers))
            ai_count = int((df["category_source"] == "AI").sum())

    mismatches = find_balance_mismatches(df)
    monthly = monthly_summary(df)
    latency = round(time.perf_counter() - start, 3)
    logger.info("Analysed %s (%s): %d rows in %.2fs", file.filename, method, len(df), latency)

    return AnalyzeResponse(
        file_name=file.filename,
        read_method=method,
        summary=Summary(**overall_summary(df)),
        spend_by_category={k: round(float(v), 2) for k, v in spend_by_category(df).items()},
        monthly={m: {"credit": float(r.credit), "debit": float(r.debit)} for m, r in monthly.iterrows()},
        balance_check=BalanceCheck(passed=mismatches.empty, mismatched_rows=to_records(mismatches)),
        ai_categorised=ai_count,
        transactions=to_records(df),
        latency_sec=latency,
    )


@app.post("/ask", response_model=AskResponse)
async def ask(
    file: UploadFile = File(..., description="Bank statement file"),
    question: str = Form(..., min_length=3, max_length=300, description="Question about the statement"),
):
    start = time.perf_counter()
    llm = get_llm()  # check the key BEFORE spending time on OCR
    df, _ = await read_statement(file)
    answer, steps = ask_statement(llm, df, question)
    latency = round(time.perf_counter() - start, 3)
    logger.info("Answered %r in %.2fs using %d tool call(s)", question, latency, len(steps))
    return AskResponse(question=question, answer=answer, steps=steps, latency_sec=latency)
