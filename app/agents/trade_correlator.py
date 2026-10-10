import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

from config import OLLAMA_MODEL_SMALL
from edgar_client import EdgarClient
from ingest import archive_url
from llm import chat
from models import TradeFinding, Transaction


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def _value_text(el, local_name):
    for child in el.iter():
        if _local(child.tag) == local_name:
            for v in child.iter():
                if _local(v.tag) == "value":
                    return (v.text or "").strip()
            return (child.text or "").strip()
    return None


def parse_form4(raw):
    root = ET.fromstring(raw)
    transactions = []
    for tx in root.iter():
        if _local(tx.tag) not in ("nonDerivativeTransaction", "derivativeTransaction"):
            continue
        shares = _value_text(tx, "transactionShares")
        price = _value_text(tx, "transactionPricePerShare")
        transactions.append(
            {
                "transaction_date": _value_text(tx, "transactionDate"),
                "code": _value_text(tx, "transactionCode"),
                "shares": float(shares) if shares else None,
                "price": float(price) if price else None,
                "acquired_disposed": _value_text(tx, "transactionAcquiredDisposedCode"),
            }
        )
    return transactions


def correlate(edgar, ticker, lookback_days=30):
    cik = edgar.resolve_ticker(ticker)
    submissions = edgar.get_submissions(cik)
    recent = submissions["filings"]["recent"]
    forms = recent["form"]

    disclosure_date = None
    for i, form in enumerate(forms):
        if form == "8-K":
            disclosure_date = recent["filingDate"][i]
            break

    start = end = None
    low = high = None
    if disclosure_date:
        end = datetime.strptime(disclosure_date, "%Y-%m-%d")
        start = end - timedelta(days=lookback_days)
        low = (start - timedelta(days=5)).strftime("%Y-%m-%d")
        high = (end + timedelta(days=2)).strftime("%Y-%m-%d")

    transactions = []
    for i, form in enumerate(forms):
        if form != "4":
            continue
        filing_date = recent["filingDate"][i]
        if low and filing_date < low:
            break
        if high and filing_date > high:
            continue
        accession = recent["accessionNumber"][i]
        document = os.path.basename(recent["primaryDocument"][i])
        raw = edgar.get_text(archive_url(cik, accession, document))
        for tx in parse_form4(raw):
            transactions.append(
                Transaction(
                    accession_number=accession,
                    transaction_date=tx["transaction_date"],
                    code=tx["code"],
                    shares=tx["shares"],
                    price=tx["price"],
                    acquired_disposed=tx["acquired_disposed"],
                )
            )

    in_window = False
    if start and end:
        for tx in transactions:
            if tx.code == "S" and tx.transaction_date:
                d = datetime.strptime(tx.transaction_date, "%Y-%m-%d")
                if start <= d <= end:
                    in_window = True
                    break

    return TradeFinding(
        ticker=ticker,
        cik=cik,
        disclosure_date=disclosure_date,
        lookback_days=lookback_days,
        transactions=transactions,
        in_window=in_window,
    )


def _in_window_sales(finding):
    if not finding.disclosure_date:
        return []
    end = datetime.strptime(finding.disclosure_date, "%Y-%m-%d")
    start = end - timedelta(days=finding.lookback_days)
    sales = []
    for tx in finding.transactions:
        if tx.code != "S" or not tx.transaction_date:
            continue
        d = datetime.strptime(tx.transaction_date, "%Y-%m-%d")
        if start <= d <= end:
            sales.append(tx)
    return sales


def summarize(finding):
    sales = _in_window_sales(finding)
    lines = [
        f"- {tx.transaction_date} sold {tx.shares} shares @ {tx.price} "
        f"(accession {tx.accession_number})"
        for tx in sales
    ]
    tx_text = "\n".join(lines) if lines else "(none)"
    prompt = (
        f"Ticker {finding.ticker} (CIK {finding.cik}). "
        f"Most recent 8-K disclosure date: {finding.disclosure_date}. "
        f"Lookback window: the {finding.lookback_days} days before that date. "
        f"Insider sales (Form 4, code S) inside the window:\n{tx_text}\n\n"
        f"Write a single factual sentence stating whether any insider sale occurred "
        f"in the window, citing the transaction dates. Do not speculate about intent."
    )
    content, model = chat(OLLAMA_MODEL_SMALL, [{"role": "user", "content": prompt}])
    return content, model


def run(edgar, ticker, lookback_days=30):
    finding = correlate(edgar, ticker, lookback_days)
    summary, model = summarize(finding)
    finding.summary = summary
    return finding, model
