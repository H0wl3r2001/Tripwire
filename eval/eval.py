import json
import os

import httpx

API_URL = os.environ.get("TRIPWIRE_API_URL", "http://app:8000")
GOLDEN_SET = os.environ.get("GOLDEN_SET", "golden_set.json")


def load_cases(path=GOLDEN_SET):
    with open(path) as f:
        data = json.load(f)
    return data["cases"]


def investigate(ticker, lookback_days):
    resp = httpx.post(
        f"{API_URL}/investigate",
        json={"ticker": ticker, "lookback_days": lookback_days},
        timeout=600.0,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    cases = load_cases()
    rows = []
    for case in cases:
        try:
            report = investigate(case["ticker"], case["lookback_days"])
            rows.append((case, report, None))
        except Exception as exc:
            rows.append((case, None, str(exc)))

    n = len(rows)
    verdict_correct = 0
    retrieval_hits = 0
    citation_valid = 0
    trade_models = {}
    filing_models = {}

    print("PER-CASE RESULTS")
    header = (
        f"{'id':<14} {'ticker':<8} {'lookback':>8} "
        f"{'expected':<18} {'actual':<18} {'verdict':<8} {'retrieval':<10} {'citations':<10}"
    )
    print(header)
    print("-" * len(header))

    for case, report, err in rows:
        if report is None:
            print(
                f"{case['id']:<14} {case['ticker']:<8} {case['lookback_days']:>8} "
                f"{case['expected_verdict']:<18} {'ERROR':<18} {'-':<8} {'-':<10} {'-':<10}"
            )
            continue

        verdict_ok = report.get("verdict") == case["expected_verdict"]
        retrieval_ok = case.get("expected_accession") in report.get("citations", [])
        citation_ok = not report.get("validation_issues")

        if verdict_ok:
            verdict_correct += 1
        if retrieval_ok:
            retrieval_hits += 1
        if citation_ok:
            citation_valid += 1

        trade_models[report.get("trade_model", "?")] = (
            trade_models.get(report.get("trade_model", "?"), 0) + 1
        )
        filing_models[report.get("filing_model", "?")] = (
            filing_models.get(report.get("filing_model", "?"), 0) + 1
        )

        print(
            f"{case['id']:<14} {case['ticker']:<8} {case['lookback_days']:>8} "
            f"{case['expected_verdict']:<18} {report.get('verdict', ''):<18} "
            f"{'OK' if verdict_ok else 'FAIL':<8} "
            f"{'OK' if retrieval_ok else 'MISS':<10} "
            f"{'OK' if citation_ok else 'FAIL':<10}"
        )

    print("\nAGGREGATE METRICS")
    print(f"correlation-flag accuracy : {verdict_correct}/{n} = {verdict_correct / n * 100:.1f}%")
    print(f"retrieval hit-rate        : {retrieval_hits}/{n} = {retrieval_hits / n * 100:.1f}%")
    print(f"citation-validity rate    : {citation_valid}/{n} = {citation_valid / n * 100:.1f}%")
    print("model routing split:")
    print(f"  trade  : {trade_models}")
    print(f"  filing : {filing_models}")


if __name__ == "__main__":
    main()
