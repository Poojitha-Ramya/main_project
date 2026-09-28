import os
from pathlib import Path
from dotenv import load_dotenv

# Ensure .env is loaded regardless of current working directory
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

from typing import List, Optional

class Config:
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT") or 8000)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_WRITER_MODEL: str = os.getenv("GROQ_WRITER_MODEL", "llama-3.1-8b-instant")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    SERPAPI_API_KEY: str = os.getenv("SERPAPI_API_KEY", "")
    DEFAULT_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

SUPPORTED_FLASH_MODELS: List[str] = [
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]

GROQ_SUPPORTED_MODELS: List[str] = [
    "llama-3.1-8b-instant",
    "llama3-8b-8192",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
]

def get_fallback_chain(primary_model: Optional[str] = None) -> List[str]:
    """
    Returns an ordered list of Gemini models to attempt with seamless redirection.
    If a model is unavailable (e.g., 503 high demand spike, 429 quota/rate limit),
    the system redirects to the next candidate model in the chain.
    """
    target = (primary_model or os.getenv("GEMINI_MODEL") or "gemini-3.6-flash").strip().lower()
    if target == "gemini-2.5-flash":
        target = "gemini-3.6-flash"

    priority_order = {
        "gemini-3.6-flash": ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash", "gemini-3.5-flash"],
        "gemini-3.7-flash": ["gemini-3.7-flash", "gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash"],
        "gemini-3.8-flash": ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash"],
        "gemini-3.5-flash": ["gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"],
        "gemini-3.5-flash-lite": ["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"],
    }

    if target in priority_order:
        return list(priority_order[target])

    chain = [target] if target else []
    for m in SUPPORTED_FLASH_MODELS:
        if m not in chain:
            chain.append(m)
    return chain

def get_groq_fallback_chain(primary_model: Optional[str] = None) -> List[str]:
    """
    Returns an ordered fallback list of Groq models specifically for WriterAgent.
    Prioritizes Llama 3.1 and Llama 3.8 models, with seamless failover to active Groq 3.8 / OSS models.
    """
    target = (primary_model or os.getenv("GROQ_WRITER_MODEL") or "llama-3.1-8b-instant").strip().lower()

    if any(k in target for k in ["3.8", "3-8b", "3 8b", "3_8", "llama3-8b"]):
        target = "llama3-8b-8192"
    elif any(k in target for k in ["3.3", "3-3", "llama-3.3"]):
        target = "llama-3.3-70b-versatile"
    elif any(k in target for k in ["3.1", "3-1", "llama-3.1", "llama3.1"]):
        if "70b" in target:
            target = "llama-3.1-70b-versatile"
        else:
            target = "llama-3.1-8b-instant"
    elif "llama" in target:
        target = "llama-3.1-8b-instant"

    priority_order = {
        "llama-3.1-8b-instant": ["llama-3.1-8b-instant", "llama3-8b-8192", "qwen/qwen3.8-27b", "openai/gpt-oss-120b"],
        "llama3-8b-8192": ["llama3-8b-8192", "llama-3.1-8b-instant", "qwen/qwen3.8-27b", "openai/gpt-oss-120b"],
        "qwen/qwen3.8-27b": ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"],
        "openai/gpt-oss-120b": ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"],
        "llama-3.3-70b-versatile": ["llama-3.3-70b-versatile", "qwen/qwen3.8-27b", "openai/gpt-oss-120b"],
        "llama-3.1-70b-versatile": ["llama-3.1-70b-versatile", "qwen/qwen3.8-27b", "openai/gpt-oss-120b"],
    }

    if target in priority_order:
        return list(priority_order[target])

    chain = [target] if target else ["llama-3.1-8b-instant"]
    for m in GROQ_SUPPORTED_MODELS:
        if m not in chain:
            chain.append(m)
    return chain

config = Config()

