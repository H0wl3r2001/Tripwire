import httpx
from fastapi import FastAPI
from pydantic import BaseModel

from config import OLLAMA_HOST, QDRANT_HOST, QDRANT_PORT
from edgar_client import EdgarClient
from graph import investigate as run_investigation
from ingest import ensure_collection, get_qdrant

app = FastAPI(title="Tripwire")

edgar = EdgarClient()
qdrant = get_qdrant()
ensure_collection(qdrant)


class InvestigateRequest(BaseModel):
    ticker: str
    lookback_days: int = 30


@app.get("/health")
async def health():
    dependencies = {}

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{OLLAMA_HOST}/")
            dependencies["ollama"] = {
                "ok": r.status_code == 200,
                "status": r.status_code,
            }
    except httpx.HTTPError as exc:
        dependencies["ollama"] = {"ok": False, "error": str(exc)}

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"http://{QDRANT_HOST}:{QDRANT_PORT}/readyz")
            dependencies["qdrant"] = {
                "ok": r.status_code == 200,
                "status": r.status_code,
            }
    except httpx.HTTPError as exc:
        dependencies["qdrant"] = {"ok": False, "error": str(exc)}

    all_ok = all(dep.get("ok") for dep in dependencies.values())
    return {"status": "ok" if all_ok else "degraded", "dependencies": dependencies}


@app.post("/investigate")
def investigate(req: InvestigateRequest):
    report = run_investigation(edgar, qdrant, req.ticker, req.lookback_days)
    return report
