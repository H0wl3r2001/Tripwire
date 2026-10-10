from typing import TypedDict

from langgraph.graph import END, StateGraph

from agents.filing_analyst import retrieve_chunks, summarize_chunks
from agents.trade_correlator import _in_window_sales, run as run_trade
from models import FilingAnalysis, Report, TradeFinding
from validation import ground_truth_accessions, llm_validate, validate_citations


class State(TypedDict, total=False):
    ticker: str
    lookback_days: int
    trade_finding: TradeFinding
    filing_analysis: FilingAnalysis
    chunks: list
    report: Report
    validation_issues: list[str]
    regeneration_count: int
    trade_model: str
    filing_model: str


def _report_text(trade, filing):
    return f"TRADE FINDING:\n{trade.summary}\n\nFILING ANALYSIS:\n{filing.summary}"


def _chunk_context(chunks):
    return "\n\n".join(
        f"accession={c['accession_number']} chunk={c['chunk_index']}: {c['text']}"
        for c in chunks
    )


def compute_verdict(trade_finding, form4_threshold):
    if trade_finding.form4_count < form4_threshold:
        return "insufficient data"
    if trade_finding.in_window:
        return "flagged"
    return "not flagged"


def build_graph(edgar, qdrant, form4_threshold=5):
    def supervisor(state):
        return {}

    def trade_node(state):
        finding, model = run_trade(edgar, state["ticker"], state["lookback_days"])
        return {"trade_finding": finding, "trade_model": model}

    def filing_node(state):
        chunks = retrieve_chunks(qdrant, state["ticker"], "material disclosure", top_k=5)
        summary, model = summarize_chunks(state["ticker"], chunks)
        analysis = FilingAnalysis(
            ticker=state["ticker"],
            accession_number=chunks[0]["accession_number"] if chunks else None,
            form_type="8-K",
            summary=summary,
            citations=sorted({c["accession_number"] for c in chunks}),
        )
        return {"filing_analysis": analysis, "chunks": chunks, "filing_model": model}

    def validate_node(state):
        trade = state["trade_finding"]
        filing = state["filing_analysis"]
        chunks = state.get("chunks", [])
        truth = ground_truth_accessions(trade, filing)
        report = _report_text(trade, filing)
        issues = [f"ungrounded accession: {a}" for a in validate_citations(report, truth)]
        llm_result = llm_validate(filing.summary, _chunk_context(chunks))
        if llm_result.strip().upper() != "VALID":
            issues.append(llm_result.strip())
        return {"validation_issues": issues}

    def regenerate_node(state):
        trade = state["trade_finding"]
        filing = state["filing_analysis"]
        chunks = state.get("chunks", [])
        truth = ground_truth_accessions(trade, filing)
        summary, _ = summarize_chunks(state["ticker"], chunks, strict_accessions=truth)
        new_filing = filing.model_copy(update={"summary": summary})
        return {
            "filing_analysis": new_filing,
            "regeneration_count": state["regeneration_count"] + 1,
        }

    def merge_node(state):
        trade = state["trade_finding"]
        filing = state["filing_analysis"]
        report = Report(
            ticker=state["ticker"],
            verdict=compute_verdict(trade, form4_threshold),
            disclosure_date=trade.disclosure_date,
            lookback_days=state["lookback_days"],
            sales=_in_window_sales(trade),
            filing_summary=filing.summary,
            citations=filing.citations,
            validation_issues=state["validation_issues"],
            trade_model=state.get("trade_model", ""),
            filing_model=state.get("filing_model", ""),
        )
        return {"report": report}

    def route(state):
        if not state["validation_issues"]:
            return "merge"
        if state["regeneration_count"] >= 1:
            return "merge"
        return "regenerate"

    graph = StateGraph(State)
    graph.add_node("supervisor", supervisor)
    graph.add_node("trade", trade_node)
    graph.add_node("filing", filing_node)
    graph.add_node("validate", validate_node)
    graph.add_node("regenerate", regenerate_node)
    graph.add_node("merge", merge_node)

    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "trade")
    graph.add_edge("supervisor", "filing")
    graph.add_edge("trade", "validate")
    graph.add_edge("filing", "validate")
    graph.add_conditional_edges(
        "validate", route, {"merge": "merge", "regenerate": "regenerate"}
    )
    graph.add_edge("regenerate", "validate")
    graph.add_edge("merge", END)

    return graph.compile()


def investigate(edgar, qdrant, ticker, lookback_days=30, form4_threshold=5):
    app = build_graph(edgar, qdrant, form4_threshold)
    initial = {
        "ticker": ticker,
        "lookback_days": lookback_days,
        "regeneration_count": 0,
        "validation_issues": [],
    }
    result = app.invoke(initial)
    return result["report"]
