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
    transactions: list[Transaction] = []
    in_window: bool = False
    summary: str = ""


class FilingAnalysis(BaseModel):
    ticker: str
    accession_number: str | None = None
    form_type: str = "8-K"
    summary: str = ""
    citations: list[str] = []
