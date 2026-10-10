# Tripwire

An agentic RAG system that cross-references SEC insider-trading filings (Form 4) against a company's material disclosures (8-K) to answer one question:

> *"For ticker X, was there insider selling in the N days before their most recent material disclosure, and what did the disclosure actually say?"*

It fuses a **structured lookup** (Form 4 transaction dates/amounts) with **unstructured retrieval** (the 8-K's actual event language), and refuses to state a connection it can't cite back to a real SEC accession number. The citation check is enforced in code, not just prompted for.

This is a research/screening tool, not a trading-signal or legal-accusation generator. Flags are framed as "worth a human's attention," never as wrongdoing.

---

## Architecture

```
                    ┌─────────────┐
  question ───────▶│  Supervisor │
                    │ (LangGraph) │
                    └──────┬──────┘
                           │ fan-out (parallel)
              ┌────────────┴────────────┐
              ▼                         ▼
     ┌─────────────────┐       ┌──────────────────┐
     │ Trade Correlator │      │  Filing Analyst  │
     │  (qwen2.5:3b)    │      │  (qwen2.5:7b)    │
     │  → data.sec.gov  │      │  → efts.sec.gov  │
     │  Form 4 lookup   │      │  → Qdrant RAG    │
     └────────┬─────────┘      └─────────┬────────┘
              │    structured finding    │  cited excerpt
              └────────────┬─────────────┘
                           ▼
                ┌────────────────────┐
                │  Citation Validator │  ← rejects & regenerates
                │   + Verdict Merge   │    any ungrounded claim
                │    (qwen2.5:7b)     │
                └──────────┬─────────┘
                           ▼
                   structured report
```

The graph is a LangGraph `StateGraph`: a supervisor fans out to the two specialists (which are independent, so they run in parallel), then routes to a validator node. If the validator finds an ungrounded claim, it regenerates the offending portion once and re-validates before merging.

### The two specialists

| Agent | Model | Job | Why that model |
|---|---|---|---|
| **Trade Correlator** | `qwen2.5:3b` | Resolve ticker→CIK, pull Form 4s, parse transaction dates, compute in-window proximity | Almost entirely deterministic date math over JSON — the LLM only *formats* the finding into a sentence |
| **Filing Analyst** | `qwen2.5:7b` | Retrieve 8-K chunks from Qdrant, summarize what was disclosed with citations | Real narrative summarization over unstructured text |

### The citation validator (the point of the project)

Two layers, both feeding a reject-and-regenerate loop:

1. **Deterministic check (enforced in code)** — every accession number cited in the report is regex-extracted and checked against the *ground-truth set* (the Form 4 accessions actually fetched + the Qdrant chunk accessions actually retrieved). A hallucinated accession cannot pass this.
2. **LLM semantic check (7b)** — flags fabricated *facts* (not just bad accessions) by comparing the summary against the retrieved chunk text.

On failure, the filing summary is regenerated with a strict prompt that permits *only* ground-truth accessions, then re-validated. The loop runs once before the report is emitted.

### Verdict logic

- `insufficient data` — fewer than 5 Form 4 filings exist for the CIK.
- `flagged` — any Form 4 **sale** (`code S`) falls inside the lookback window before the disclosure date.
- `not flagged` — otherwise.

---

## Tech stack

| Component | Choice |
|---|---|
| Orchestration | LangGraph (supervisor + 2 typed specialists + validator) |
| LLM inference | Ollama, CPU-only — `qwen2.5:3b` (structured) + `qwen2.5:7b` (narrative + validation) |
| Embeddings | Ollama `nomic-embed-text` (768-dim) |
| Vector store | Qdrant (Docker) — dense + sparse (BM25) vectors, RRF fusion |
| Structured storage | SQLite (EDGAR response cache) |
| Data | SEC EDGAR (`data.sec.gov`, `efts.sec.gov`, `www.sec.gov`) |
| API | FastAPI + Uvicorn |
| Data contracts | Pydantic models on every agent handoff |

---

## Design tradeoffs

**Why two routed local models instead of one model (or a cloud gateway).**

The two specialists have genuinely different needs. The Trade Correlator is ~90% deterministic date arithmetic over structured JSON — the LLM's only job is to turn a `TradeFinding` into a sentence. Routing it to a 3B model instead of the 7B model is a real, *measurable* cost win (the eval reports the routing split), and it turns "model routing" from a slide into working code. The Filing Analyst and the validator do actual narrative reasoning, so they get the 7B.

I chose local CPU-only Ollama over a cloud gateway because the project needs **no paid API key** and matches the existing hardware constraint (AMD GPU passthrough into WSL2 Docker is unreliable as of mid-2026). The tradeoff is: single-node CPU inference is slow (seconds per generation) and doesn't scale — that's the first thing to move in production (see below).

**Why two agents and not four.** The correlation claim only needs two data shapes — structured Form 4 dates and unstructured 8-K text. A third or fourth agent (e.g., a separate "date-math" agent, a separate "retriever" agent) would add handoff surface area without adding signal. The interesting engineering is the *reconciliation* between the two, not the number of specialists.

**Why SQLite and not Postgres.** At solo-project scale, one less container and zero operational overhead beats a real database. The EDGAR cache is the only thing SQLite holds, and it's a write-once-read-many cache. This is the second thing to move in production.

---

## Getting started

### Prerequisites

- Docker + Docker Compose.
- No paid API keys. The only requirement is a compliant SEC `User-Agent`.

### 1. Configure

```bash
cp .env.example .env
# edit .env: set SEC_USER_AGENT to "YourName your@email.com" (your real identity)
```

### 2. Pull the models (one-time)

```bash
docker compose up -d ollama qdrant
docker compose exec ollama ollama pull qwen2.5:3b
docker compose exec ollama ollama pull qwen2.5:7b
docker compose exec ollama ollama pull nomic-embed-text
```

### 3. Start the stack

```bash
docker compose up --build -d
```

### 4. Ingest a filing (per ticker)

The filing analyst retrieves from Qdrant, so a ticker must be ingested before `/investigate` can cite its 8-K. The demo ingests AAPL's latest 8-K:

```bash
docker compose run --rm --no-deps app python ingest.py
```

To ingest another ticker:

```bash
docker compose run --rm --no-deps app python -c "
from edgar_client import EdgarClient
from ingest import get_qdrant, ensure_collection, ingest_filing
c = EdgarClient(); q = get_qdrant(); ensure_collection(q)
print(ingest_filing(c, q, 'MSFT', '8-K'))
c.close()
"
```

---

## API

### `GET /health`

Pings Ollama and Qdrant.

```bash
curl -s http://localhost:8000/health
```

### `POST /investigate`

```bash
curl -s -X POST http://localhost:8000/investigate \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "lookback_days": 30}'
```

Request: `{"ticker": str, "lookback_days": int}` (default 30).

Response (a `Report`):

```json
{
  "ticker": "AAPL",
  "verdict": "not flagged",
  "disclosure_date": "2026-07-30",
  "lookback_days": 30,
  "sales": [],
  "filing_summary": "Apple reported its fiscal 2026 third quarter results…",
  "citations": ["0000320193-26-000018"],
  "validation_issues": [],
  "trade_model": "qwen2.5:3b",
  "filing_model": "qwen2.5:7b"
}
```

`verdict` is one of `flagged`, `not flagged`, `insufficient data`. `validation_issues` is empty when every claim traces to a real accession number.

---

## Evaluation

`eval/golden_set.json` holds hand-verified `(ticker, lookback_days, expected_verdict, expected_accession)` cases. `eval/eval.py` calls `/investigate` for each and reports four numbers:

- **correlation-flag accuracy** — verdict vs. ground truth.
- **retrieval hit-rate** — did the right 8-K accession get retrieved.
- **citation-validity rate** — target 100% (any ungrounded claim is an automatic fail).
- **model routing split** — which model handled which sub-agent call.

Run it via an ephemeral container (no local venv, nothing installed on the host):

```bash
docker run --rm \
  --network tripwire_default \
  -v "$(pwd)/eval:/eval" -w /eval \
  -v eval_pip_cache:/root/.cache/pip \
  python:3.12-slim \
  bash -c "pip install -q -r requirements.txt && python eval.py"
```

> The golden set must be hand-verified against EDGAR directly. A golden set generated by the same model you're evaluating measures nothing.

---

## Scalability & production readiness

This is a single-node, CPU-only demo. Here is what breaks first at real volume, in order, and the production fix for each — described, not built.

### 1. Single-node CPU Ollama under concurrent queries

**What breaks:** each `/investigate` makes 2–4 blocking LLM calls (trade summary, filing summary, validation, possibly regeneration). On CPU, a single 7B generation takes seconds; concurrent requests queue and time out. The graph also runs synchronously inside the request handler.

**Production fix:** move inference to a GPU-backed or hosted endpoint (or a dedicated inference service with a queue), and run the graph asynchronously behind a job queue (e.g., Celery/RQ) with a `202 Accepted` + polling/status endpoint instead of a blocking HTTP call.

### 2. SQLite under concurrent writes

**What breaks:** the EDGAR response cache is a single SQLite file. Concurrent `/investigate` calls (and concurrent eval runs) contend on writes; SQLite serializes writers and will become the bottleneck, and it doesn't shard.

**Production fix:** replace the cache with a real store — Postgres (or Redis for a pure cache) — and add a shared, distributed rate limiter so the 10 req/s SEC fair-access limit is enforced across all workers, not per-process.

### 3. Qdrant single-container

**What breaks:** one Qdrant node holds all vectors in memory/disk; it doesn't scale horizontally and is a single point of failure.

**Production fix:** run Qdrant in distributed mode (multiple shards/replicas), or move to a managed vector store.

### 4. On-demand ingestion

**What breaks:** ingestion is manual and per-ticker. There's no scheduler, no incremental updates, and no coverage beyond the tickers you remember to ingest.

**Production fix:** a scheduled/batch ingestion pipeline that watches the watchlist, fetches new 8-K/10-K filings, chunks, embeds, and upserts — with idempotent point IDs (already UUID5-keyed) so re-runs are safe.

### 5. Observability

**What breaks:** there's no tracing, metrics, or alerting. When a citation-validation failure or a slow LLM call happens, you can't see where.

**Production fix:** OpenTelemetry tracing across the graph nodes, Prometheus metrics (per-agent latency, validation-failure rate, routing split), and structured logs. This was deliberately excluded from the demo (see the brief §4) — it's the right call at team scale, not solo CPU scale.

### 6. The watchlist

**What breaks:** the demo scopes to a fixed watchlist of well-known tickers. There's no coverage of the full EDGAR filer universe, and no options/derivatives data (Form 4 equity transactions only).

**Production fix:** expand the watchlist and add a ticker→CIK resolution cache; add Form 4 derivative-table handling if options matter.

---

## Project structure

```
Tripwire/
├── docker-compose.yml
├── .env.example
├── README.md
├── app/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                 # FastAPI: /health, /investigate
│   ├── config.py               # env-driven config
│   ├── edgar_client.py         # rate-limited, cached SEC EDGAR client
│   ├── ingest.py               # chunk + embed + upsert into Qdrant (dense + BM25)
│   ├── graph.py                # LangGraph supervisor + specialists + validator
│   ├── validation.py           # citation checking + regeneration
│   ├── models.py               # Pydantic schemas for every handoff
│   ├── llm.py                  # Ollama chat helper
│   └── agents/
│       ├── trade_correlator.py # Form 4 lookup (qwen2.5:3b)
│       └── filing_analyst.py   # 8-K retrieval + summary (qwen2.5:7b)
├── eval/
│   ├── eval.py
│   ├── requirements.txt
│   └── golden_set.json
└── tests/
    └── test_api.py
```

---

## Non-goals

- No options/derivatives data — Form 4 equity transactions only.
- No real-time streaming ingestion — on-demand/batch fetch is enough to prove the pattern.
- No coverage of all 500k+ EDGAR filers — a fixed watchlist of well-known tickers.
- No fine-tuning, no GPU passthrough.
- No Postgres/Prometheus/Grafana as built infrastructure — described as the upgrade path above.
- No investment or legal advice framing anywhere.

---

## Notes

- **SEC fair access:** every request carries a descriptive `User-Agent` (from `SEC_USER_AGENT` in `.env`) and is rate-limited to ≤10 req/s. Responses are cached in SQLite so re-running the eval doesn't re-hit EDGAR.
- **Outbound network:** the `app` container reaches exactly three hosts — `data.sec.gov`, `efts.sec.gov`, and `www.sec.gov` (filing document downloads). Don't add calls to other domains without noting why here.
- **No secrets:** `.env` (real values) is gitignored; `.env.example` (placeholders) is committed. This project needs no paid API key.
