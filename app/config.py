import os


SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "")
OLLAMA_MODEL_SMALL = os.getenv("OLLAMA_MODEL_SMALL", "qwen2.5:3b")
OLLAMA_MODEL_LARGE = os.getenv("OLLAMA_MODEL_LARGE", "qwen2.5:7b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
