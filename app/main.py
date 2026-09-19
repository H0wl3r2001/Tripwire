import httpx
from fastapi import FastAPI

from config import OLLAMA_HOST, QDRANT_HOST, QDRANT_PORT

app = FastAPI(title="Tripwire")


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


@app.get("/investigate")
async def investigate():
    return {"status": "not_implemented"}
