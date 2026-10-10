import re

from config import OLLAMA_MODEL_LARGE
from llm import chat

ACCESSION_RE = re.compile(r"\d{10}-\d{2}-\d{6}")


def extract_accessions(text):
    return set(ACCESSION_RE.findall(text))


def ground_truth_accessions(trade_finding, filing_analysis):
    truth = {tx.accession_number for tx in trade_finding.transactions}
    truth.update(filing_analysis.citations)
    return truth


def validate_citations(report_text, ground_truth):
    cited = extract_accessions(report_text)
    return sorted(cited - ground_truth)


def llm_validate(report_text, context, model=OLLAMA_MODEL_LARGE):
    prompt = (
        "You are a strict citation validator. The report below must be grounded "
        "entirely in the provided context, which is the only data that was actually "
        "fetched. Identify any factual claim in the report that is not supported by "
        "the context. Respond with exactly 'VALID' if every claim is supported. "
        "Otherwise respond with 'INVALID' and list each unsupported claim on its own "
        "line, prefixed with '- '.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"REPORT:\n{report_text}"
    )
    content, _ = chat(model, [{"role": "user", "content": prompt}])
    return content
