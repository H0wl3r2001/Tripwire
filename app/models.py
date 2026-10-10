from pydantic import BaseModel


class Transaction(BaseModel):
    accession_number: str
    transaction_date: str | None = None
    code: str | None = None
    shares: float | None = None
    price: float | None = None
    acquired_disposed: str | None = None


class TradeFinding(BaseModel):
    ticker: str
    cik: str
    disclosure_date: str | None = None
    lookback_days: int
    form4_count: int = 0
    transactions: list[Transaction] = []
    in_window: bool = False
    summary: str = ""


class FilingAnalysis(BaseModel):
    ticker: str
    accession_number: str | None = None
    form_type: str = "8-K"
    summary: str = ""
    citations: list[str] = []


class Report(BaseModel):
    ticker: str
    verdict: str
    disclosure_date: str | None = None
    lookback_days: int
    sales: list[Transaction] = []
    filing_summary: str = ""
    citations: list[str] = []
    validation_issues: list[str] = []
