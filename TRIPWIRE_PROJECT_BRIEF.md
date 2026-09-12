# Project Brief: Tripwire — Agentic Filing/Trade Correlation RAG

Hand this file to opencode inside an empty project folder (`~/projects/tripwire`, not `/mnt/c/...`) as the starting instruction.

## 1. The Problem

Public companies disclose material events via 8-K filings. Company insiders (executives, directors, 10%+ owners) disclose their own stock trades via Form 4. Both are public record on SEC EDGAR. But nobody manually cross-references them — did an insider sell shares in the days before their company disclosed something bad? Answering that today means pulling two different filing types by hand and eyeballing dates.

**Tripwire** answers a question like *"For ticker X, was there insider selling in the N days before their most recent material disclosure, and what did the disclosure actually say?"* It fuses a structured lookup (Form 4 transaction dates/amounts) with unstructured retrieval (the 8-K's actual risk/event language), and refuses to state a connection it can't cite back to a real accession number.

This is a research/screening tool, not a trading-signal or legal-accusation generator — the README should frame flags as "worth a human's attention," not wrongdoing. Pick eval tickers accordingly (large, frequently-covered companies where any real pattern found is already public knowledge, e.g., from major indices).

## 2. Why This Shape, Not a Simpler One

A single RAG-over-8-Ks system would be a demo. What makes this a portfolio piece is that the correlation claim requires **two genuinely different data shapes done right** — precise date arithmetic over structured API JSON, and grounded narrative retrieval over filing text — reconciled by an orchestrator that won't let either agent's claim stand unless it traces to a real filing. That reconciliation step, not the retrieval, is the interesting engineering.

## 3. Success Criteria (Definition of Done)

- `docker-compose up` starts `qdrant`, `ollama`, `app` with no manual steps beyond pulling both models into Ollama (documented one-time step).
- An `/investigate` endpoint accepts `{"ticker": str, "lookback_days": int}` (default 30) and returns: a timeline of Form 4 sales in that window, the retrieved 8-K excerpt that followed, a verdict (`flagged` / `not flagged` / `insufficient data`), and a citation (accession number, or Form 4 accession + transaction row) for every factual claim in the response.
- A response is **rejected and regenerated** if any claim can't be traced to a real fetched accession number — this is enforced in code, not just prompted for.
- `eval/eval.py` runs a hand-verified 25-case golden set and reports: correlation-flag accuracy against ground truth, retrieval hit-rate (did the right 8-K section get retrieved), citation-validity rate (target 100% — any ungrounded claim is an automatic fail), and the cheap-vs-expensive model routing split.
- `README.md`: architecture, one real design tradeoff you can defend out loud, a "how this would need to change past a demo" scalability section, and sample `curl` commands.
- No hardcoded secrets. SEC's fair-access policy requires a descriptive `User-Agent` (app name + contact email) on every request — that's config, not a secret, but it still lives in `.env`, not in code.

## 4. Tech Stack

| Component | Choice | Why |
|---|---|---|
| Orchestration | LangGraph — a 2-specialist + supervisor graph, typed Pydantic state at every handoff | Actively-requested skill in current AI eng postings; a hand-rolled state machine would work too but LangGraph is worth the practice here |
| LLM inference | Ollama, **CPU-only**, two models: `qwen2.5:3b` for the structured Trade-Correlator agent (mostly date math over JSON, doesn't need a big model), `qwen2.5:7b` for the Filing-Analyst agent and the citation-validation pass | Matches your existing hardware constraint (RX 9060 XT GPU passthrough into Docker on WSL2 isn't reliable enough to depend on yet — CPU-only, same call as before). Using two sizes turns "model routing" from a slide into working code, without needing a second paid provider. |
| Embeddings | Ollama `nomic-embed-text` | One runtime for LLM + embeddings — avoids pulling in a second heavy stack (sentence-transformers/PyTorch) for a project that's already got enough moving parts |
| Vector store | Qdrant (Docker) | Same as your last project — no reason to switch conventions |
| Structured storage | SQLite — Form 4/8-K cache, per-run trace logs, eval results | Postgres would be overkill at this scale for a solo project; SQLite is one less container |
| Real-world data | SEC EDGAR (`data.sec.gov`, `efts.sec.gov`) | Free, no API key, no signup — only requirement is a compliant `User-Agent` header (your name/app + contact email) and staying under 10 requests/second. Verified current as of mid-2026. |
| API layer | FastAPI + Uvicorn | Consistent with prior project |
| Data contracts | Pydantic models on every agent handoff | Typed handoffs instead of freeform text between agents — catches malformed data at the boundary instead of three steps downstream |

**Explicitly not used:** Postgres, Prometheus, Grafana, OpenTelemetry SDK. A full observability stack is the right call at team scale, not for a solo CPU-only demo — see the scalability section required in the README (§7) for where that line actually moves.

## 5. Architecture

```
                    ┌─────────────┐
  question ────────▶│  Supervisor  │
                    │  (LangGraph) │
                    └──────┬───────┘
                           │ fan-out
              ┌────────────┴────────────┐
              ▼                         ▼
     ┌─────────────────┐      ┌──────────────────┐
     │ Trade Correlator │      │  Filing Analyst   │
     │  (qwen2.5:3b)    │      │  (qwen2.5:7b)     │
     │  → data.sec.gov  │      │  → efts.sec.gov   │
     │  Form 4 lookup   │      │  → Qdrant RAG     │
     └────────┬─────────┘      └─────────┬────────┘
              │  structured finding       │  cited excerpt
              └────────────┬───────────────┘
                            ▼
                 ┌────────────────────┐
                 │  Citation Validator │  ← rejects & regenerates
                 │   + Verdict Merge   │    any ungrounded claim
                 │    (qwen2.5:7b)     │
                 └──────────┬─────────┘
                            ▼
                    structured report
```

## 6. Folder Structure

```
tripwire/
├── docker-compose.yml
├── .env.example              # SEC_USER_AGENT, OLLAMA_MODEL_SMALL, OLLAMA_MODEL_LARGE, QDRANT_HOST/PORT
├── README.md
├── TRIPWIRE_PROJECT_BRIEF.md
├── .gitignore                # see required entries below
├── app/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                 # FastAPI app, /investigate, /health
│   ├── config.py
│   ├── edgar_client.py         # rate-limited (10 req/s), cached in SQLite, User-Agent enforced
│   ├── graph.py                 # LangGraph: supervisor + 2 specialists + validator node
│   ├── agents/
│   │   ├── trade_correlator.py
│   │   └── filing_analyst.py
│   ├── ingest.py                 # chunk + embed 8-K/10-K text into Qdrant
│   ├── validation.py              # citation-checking against fetched accession numbers
│   └── models.py                   # Pydantic schemas for every handoff
├── eval/
│   ├── eval.py                       # runs golden_set.json against the running API
│   ├── requirements.txt
│   └── golden_set.json                # 25 hand-verified (ticker, window, expected verdict, expected citation) cases
└── tests/
    └── test_api.py
```

Only `app/` gets a Dockerfile — `qdrant` and `ollama` run from official images. `eval/` and `tests/` don't get a Dockerfile or compose service either, but **don't reach for a local venv** — a correction from how the last project's brief was originally written: creating a venv still depends on `python3`/`python3-venv` already existing on the WSL host outside any project, which conflicts with this environment's "don't install runtimes on the base system" rule. Use the same ephemeral-container pattern the RAG project actually shipped with instead:

```bash
docker run --rm \
  --network tripwire_default \
  -v "$(pwd)/eval:/eval" -w /eval \
  -v eval_pip_cache:/root/.cache/pip \
  python:3.12-slim \
  bash -c "pip install -q -r requirements.txt && python eval.py"
```

(Same pattern for `tests/test_api.py`.) `eval_pip_cache` is an optional named volume that just speeds up repeat runs. Document this exact command in the README so it's copy-pasteable.

Two named Docker volumes are required in `docker-compose.yml` — `ollama_data:/root/.ollama` and `qdrant_data:/qdrant/storage` — so both pulled models and ingested filings survive `docker-compose down`.

`.gitignore` must include, at minimum: `.env` (only `.env.example` is committed), `*.db` / `*.sqlite3` (the EDGAR response cache, trace logs, and eval results all live in SQLite files that shouldn't be tracked), `__pycache__/`, and `*.pyc`. Create it in Phase 1, before any other file that might contain a secret or a generated artifact.

## 7. Implementation Phases

**Stop after each phase and wait for explicit confirmation before starting the next one.** Same discipline as the RAG project: summarize what changed, give the exact `docker-compose`/`curl` commands to verify it, and wait. This project has more moving parts than the last one (two agents, real outbound network calls to a third party) — the per-phase checkpoint matters more here, not less.

### Phase 1 — Scaffolding
- Folder structure above. `docker-compose.yml` with `qdrant`, `ollama` (CPU-only, no GPU device reservation), `app`, both named volumes. Directly above the `ollama` service definition, add the same kind of guard comment the RAG project uses, so a later edit doesn't "fix" what looks like a missing GPU reservation:
  ```yaml
  # CPU-only by design: AMD GPU passthrough (Vulkan/ROCm) into a WSL2 Docker
  # container is unreliable as of mid-2026 (missing /dev/dri, driver mismatches,
  # ERROR_INCOMPATIBLE_DRIVER). Do not add a `deploy.resources.reservations.devices`
  # GPU block here without re-validating passthrough works first — see
  # TRIPWIRE_PROJECT_BRIEF.md section 4.
  ```
- `.env.example` including `SEC_USER_AGENT` (format: `"YourName your@email.com"` — required or EDGAR returns 403).
- `/health` endpoint pinging `ollama` and `qdrant`.

### Phase 2 — EDGAR Client
- `edgar_client.py`: wraps `data.sec.gov/submissions/CIK##########.json` (filing history, including Form 4s), `data.sec.gov` XBRL endpoints, and `efts.sec.gov` full-text search — all with the required `User-Agent` header from `.env`.
- Token-bucket rate limiter enforcing ≤10 req/s. SQLite response cache keyed on URL, so re-running the eval set doesn't re-hit EDGAR every time.
- Ticker → CIK resolution (EDGAR indexes by CIK, not ticker).

### Phase 3 — Ingestion + Retrieval
- `ingest.py`: fetch an 8-K/10-K's actual document text, chunk it, embed with `nomic-embed-text`, upsert into Qdrant with `{accession_number, ticker, form_type, chunk_index, text}` metadata.
- Standard top-k retrieval, same pattern as the earlier RAG project.

### Phase 4 — The Two Agents
- `trade_correlator.py`: given a ticker + window, pulls Form 4 filings from the cached EDGAR client, computes date proximity to the target disclosure, returns a structured `TradeFinding` (transactions, dates, whether any fall in-window) — this agent barely needs the LLM at all beyond formatting, which is exactly why it's routed to the 3b model.
- `filing_analyst.py`: retrieves the relevant 8-K chunk(s) from Qdrant, asks the 7b model to summarize what was disclosed, cites the accession number and chunk.

### Phase 5 — Supervisor, Validation, Merge
- `graph.py`: LangGraph state graph — supervisor dispatches both specialists (parallel, since they're independent), then routes to a validator node.
- `validation.py`: for every claim in the merged output, check it resolves to either a real Form 4 transaction row already fetched, or a Qdrant chunk with a real accession number. If any claim fails, regenerate that portion — don't just flag it, actually loop back once.
- Verdict logic: `flagged` if any Form 4 sale falls inside the lookback window before the disclosure date; `insufficient data` if fewer than N Form 4 filings exist for that CIK; `not flagged` otherwise.

### Phase 6 — Evaluation Harness
- `golden_set.json`: 25 cases, each a real `(ticker, lookback_days, expected_verdict)` you've manually verified against EDGAR directly — this is the part that takes real time, don't shortcut it, the whole credibility of the eval rests on it being hand-checked, not model-generated.
- `eval/eval.py`: calls `/investigate` for each case, scores correlation-flag accuracy against your verified ground truth, retrieval hit-rate, citation-validity rate, and reports which model handled which sub-agent call (routing distribution). Print a summary table.
- Run `eval/eval.py` (and `tests/test_api.py`) via the ephemeral `docker run` command in §6 — no local venv, nothing installed on the WSL host.

Once real numbers come back from this phase, treat any further retrieval or agent-quality fixes the way the RAG project's Phase 7 was scoped: diagnose the *specific* failure category the eval numbers actually show (e.g. low citation-validity vs. low correlation-flag accuracy are different bugs with different fixes) before picking a fix, and write down what you're explicitly deferring and why — don't default to generic polish.

### Phase 7 — Polish
- `README.md`: architecture (reuse §5 diagram), the design tradeoff you'd actually defend (e.g., "why 2 agents and not 4," "why routed local models instead of a cloud gateway"), and a **scalability section** — name specifically what breaks first at real volume (SQLite under concurrent writes, single-node CPU Ollama under concurrent queries, Qdrant single-container) and what the production fix would look like, without building it.
- Sample `curl` commands against `/investigate`.

## 8. Non-Goals

- No options/derivatives data — Form 4 equity transactions only.
- No real-time streaming ingestion — on-demand/batch fetch against EDGAR is enough to prove the pattern.
- No coverage of all 500k+ EDGAR filers — scope the eval set and demo to a fixed watchlist of 10-20 well-known tickers.
- No fine-tuning, no GPU passthrough (see §4).
- No Postgres/Prometheus/Grafana as built infrastructure — described as the upgrade path in the README instead.
- No investment or legal advice framing anywhere in the UI or README copy.

## 9. Notes for the Agent

- Favor small, testable commits per phase.
- The citation-validation step in Phase 5 is the actual point of this project — don't let it become an afterthought bolted on at the end.
- Hand-verify the golden set yourself against EDGAR directly before trusting the eval numbers; a golden set generated by the same model you're evaluating measures nothing.
- Scope discipline: all file writes, builds, and container operations stay inside this project directory (`~/projects/tripwire`). Do not read, write, or mount anything outside it — no `~/.ssh`, `~/.aws`, `~/.azure`, other project folders, or Windows paths under `/mnt/c` other than this one if it happens to be cloned there.
- Do not run containers with `--privileged`, extra bind mounts to the host filesystem beyond the two named volumes and the project's own directories, or host networking mode. Services communicate over the default compose network.
- Outbound network access from `app` is expected to reach exactly three hosts: `data.sec.gov`, `efts.sec.gov`, and `www.sec.gov` (filing document downloads). This is the first project in this environment that legitimately needs live internet access at all — that's a wider blast radius than the fully-offline RAG project, not a smaller one, so don't casually add calls to other domains without noting why in the README.
- Never write a real secret or credential into any tracked file. `.env` (real values, including your actual contact email in `SEC_USER_AGENT`) is gitignored; `.env.example` (placeholder values only) is committed. This project needs no paid API key to run — if a task ever seems to require one, stop and ask rather than inventing a workaround.

---

## 10. How This Maps to Current AI Engineering JDs

| JD language | Where Tripwire demonstrates it |
|---|---|
| "Experience building RAG pipelines" | Chunking, embedding, retrieval over real 8-K/10-K text |
| "Agent orchestration / agentic workflows (LangGraph)" | Supervisor + 2 typed specialists, not a single-agent wrapper |
| "Eval-driven development / offline evaluation" | Hand-verified golden set scored on 3 separate dimensions, not vibes |
| "LLM cost/latency optimization" | Working complexity-based model router (3b vs 7b), measured routing split |
| "Grounding / hallucination mitigation" | Citation validation that rejects and regenerates ungrounded claims — enforced in code |
| "Integrating third-party/public data sources" | Real EDGAR API, rate-limiting, caching, not a static dataset |
| "Production concerns: caching, retries, observability" | SQLite response cache, rate limiter, per-run trace logs |

## 11. Resume Line (fill in real numbers after building — don't estimate them now)

*"Built a multi-agent RAG system that cross-references SEC insider-trading filings against material company disclosures, reaching **X%** correlation-flag accuracy and **100%** citation-validity against a 25-case hand-verified eval set, with a complexity-based local model router that handled **Y%** of sub-agent calls on a 3B model instead of defaulting everything to the larger 7B."*

Lead with that number, not the stack, the same way the source material you compared this against argues for — but the number has to come from your actual eval run, not from this brief.