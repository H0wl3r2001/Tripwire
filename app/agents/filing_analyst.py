from config import OLLAMA_MODEL_LARGE
from ingest import hybrid_retrieve
from llm import chat
from models import FilingAnalysis


def retrieve_chunks(qdrant, ticker, query, top_k=5):
    return hybrid_retrieve(qdrant, query, ticker=ticker, top_k=top_k)


def summarize_chunks(ticker, chunks, strict_accessions=None):
    if not chunks:
        return "", OLLAMA_MODEL_LARGE
    context = "\n\n".join(
        f"[{i}] accession={c['accession_number']} chunk={c['chunk_index']}:\n{c['text']}"
        for i, c in enumerate(chunks)
    )
    if strict_accessions:
        allowed = ", ".join(sorted(strict_accessions))
        prompt = (
            f"Summarize what was disclosed in the following 8-K excerpts for {ticker}. "
            f"You may cite ONLY these accession numbers: {allowed}. "
            f"Do not cite any other accession number. "
            f"Do not invent facts that are not present in the excerpts.\n\n{context}"
        )
    else:
        prompt = (
            f"Summarize what was disclosed in the following 8-K excerpts for {ticker}. "
            f"For every factual claim, cite the accession number in brackets. "
            f"Do not invent facts that are not present in the excerpts.\n\n{context}"
        )
    content, model = chat(OLLAMA_MODEL_LARGE, [{"role": "user", "content": prompt}])
    return content, model


def analyze(qdrant, ticker, query, top_k=5):
    chunks = retrieve_chunks(qdrant, ticker, query, top_k=top_k)
    content, model = summarize_chunks(ticker, chunks)
    return (
        FilingAnalysis(
            ticker=ticker,
            accession_number=chunks[0]["accession_number"] if chunks else None,
            form_type="8-K",
            summary=content,
            citations=sorted({c["accession_number"] for c in chunks}),
        ),
        model,
    )
