import httpx

from config import OLLAMA_HOST


def chat(model, messages, temperature=0.0):
    resp = httpx.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=300.0,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"], data.get("model", model)
