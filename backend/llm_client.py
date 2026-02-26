"""
llm_client.py — LLM Integration Layer
Attempts Ollama (local) first, falls back to OpenAI if unavailable.
"""

import os
import httpx
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
REQUEST_TIMEOUT = 180.0  # 3 min — covers cold-start model loading


# ── Ollama ─────────────────────────────────────────────────────────────────────

async def _check_ollama_available() -> bool:
    """Quick health check — returns True if Ollama is reachable."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            return resp.status_code == 200
    except Exception:
        return False


async def _call_ollama(messages: list[dict]) -> str:
    """Send a chat request to the local Ollama instance."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
        }
    }
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]


# ── OpenAI ─────────────────────────────────────────────────────────────────────

async def _call_openai(messages: list[dict]) -> str:
    """Send a chat request to the OpenAI API."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set — cannot fall back to OpenAI.")

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 800,
    }
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


# ── Public Entry Point ─────────────────────────────────────────────────────────

async def get_llm_response(
    messages: list[dict],
    force_provider: Optional[str] = None
) -> tuple[str, str, str]:
    """
    Returns (response_text, provider, model_name).
    provider is one of: 'ollama', 'openai', 'error'
    """
    # Allow forced provider for testing
    if force_provider == "openai":
        try:
            text = await _call_openai(messages)
            return text, "openai", OPENAI_MODEL
        except Exception as e:
            logger.error(f"OpenAI call failed: {e}")
            return f"Error: {str(e)}", "error", "none"

    # Try Ollama first
    ollama_available = await _check_ollama_available()
    if ollama_available:
        try:
            text = await _call_ollama(messages)
            logger.info(f"Response from Ollama ({OLLAMA_MODEL})")
            return text, "ollama", OLLAMA_MODEL
        except Exception as e:
            logger.warning(f"Ollama call failed despite health check passing: {e}")

    # Fallback to OpenAI
    if OPENAI_API_KEY:
        try:
            text = await _call_openai(messages)
            logger.info(f"Response from OpenAI ({OPENAI_MODEL})")
            return text, "openai", OPENAI_MODEL
        except Exception as e:
            logger.error(f"OpenAI fallback also failed: {e}")
            return (
                "I'm sorry, I'm unable to connect to the AI service right now. "
                "Please ensure Ollama is running locally (`ollama serve`) or that "
                "a valid OPENAI_API_KEY is configured.",
                "error",
                "none",
            )

    return (
        "No AI provider is available. Please start Ollama (`ollama serve && ollama pull mistral`) "
        "or set the OPENAI_API_KEY environment variable.",
        "error",
        "none",
    )
