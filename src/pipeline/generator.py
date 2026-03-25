"""
src/pipeline/generator.py — LLM Answer Generation
===================================================
Supports two providers based on config.yaml → llm.provider:
  - "gemini"  : Google Gemini API (default, recommended — no local GPU needed)
  - "ollama"  : Local Ollama/Mistral (requires Ollama running locally)

Gemini setup:
  Set env variable: GEMINI_API_KEY=your_key_here
  Or set config.yaml → llm.gemini_api_key (not recommended for production)

Usage:
    from src.pipeline.generator import generate_answer
    answer = generate_answer(question, context_chunks, config)
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    try:
        return yaml.safe_load(Path("config.yaml").read_text())
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Prompt builder (shared by both providers)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are MediRAG, a medical AI assistant. "
    "Answer the question using ONLY the provided context. "
    "Be concise, accurate, and cite specific details from the context. "
    "If the context does not contain enough information, say so explicitly. "
    "Do NOT invent facts, dosages, or clinical recommendations not present in the context."
)


def _build_prompt(question: str, context_chunks: list[dict]) -> str:
    """Build the RAG prompt from the question + retrieved chunks."""
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        text = chunk.get("text") or chunk.get("chunk_text", "")
        source = chunk.get("source", "")
        pub_type = chunk.get("pub_type", "")
        header = f"[Source {i}"
        if pub_type:
            header += f" | {pub_type}"
        if source:
            header += f" | {source}"
        header += "]"
        context_parts.append(f"{header}\n{text.strip()}")

    context_block = "\n\n".join(context_parts)
    return (
        f"{_SYSTEM_PROMPT}\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER (based only on the context above):"
    )


# ---------------------------------------------------------------------------
# Gemini provider
# ---------------------------------------------------------------------------

def _generate_gemini(prompt: str, config: dict) -> str:
    llm_cfg = config.get("llm", {})

    # Try env var first, then .env file, then config
    api_key = os.environ.get("GEMINI_API_KEY") or llm_cfg.get("gemini_api_key")
    if not api_key:
        # Try loading from .env file if present
        env_file = Path(".env")
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("GEMINI_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break

    if not api_key:
        raise RuntimeError(
            "Gemini API key not found. "
            "Either: (1) set GEMINI_API_KEY=your_key in the same terminal as uvicorn, "
            "or (2) create a .env file with GEMINI_API_KEY=your_key in the project root."
        )

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise RuntimeError(
            "google-genai not installed. Run: pip install google-genai"
        )

    model_name = llm_cfg.get("gemini_model", "gemini-2.0-flash")
    client = genai.Client(api_key=api_key)

    logger.info("Calling Gemini API (model=%s)...", model_name)
    t0 = time.perf_counter()

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=float(llm_cfg.get("generation_temperature", 0.7)),
                max_output_tokens=512,
            ),
        )
    except Exception as exc:
        raise RuntimeError(f"Gemini API error: {exc}") from exc

    elapsed = int((time.perf_counter() - t0) * 1000)
    answer = response.text.strip() if response.text else ""

    if not answer:
        raise RuntimeError("Gemini returned an empty response.")

    logger.info("Gemini generated answer in %d ms (%d chars)", elapsed, len(answer))
    return answer


# ---------------------------------------------------------------------------
# Ollama provider (kept as fallback)
# ---------------------------------------------------------------------------

def _generate_ollama(prompt: str, config: dict) -> str:
    import requests as _requests

    llm_cfg = config.get("llm", {})
    base_url = llm_cfg.get("base_url", "http://localhost:8080")
    model = llm_cfg.get("model", "mistral")
    timeout = llm_cfg.get("timeout_seconds", 120)
    temperature = llm_cfg.get("generation_temperature", 0.7)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 512},
    }

    url = f"{base_url}/api/generate"
    logger.info("Calling Ollama (%s @ %s)...", model, base_url)
    t0 = time.perf_counter()

    try:
        resp = _requests.post(url, json=payload, timeout=timeout)
    except _requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Ollama is not running at {base_url}. Start with: ollama serve"
        ) from exc
    except _requests.exceptions.Timeout as exc:
        raise RuntimeError(
            f"Ollama timed out after {timeout}s. Increase llm.timeout_seconds in config.yaml."
        ) from exc

    if resp.status_code != 200:
        raise RuntimeError(f"Ollama HTTP {resp.status_code}: {resp.text[:300]}")

    try:
        data = resp.json()
        answer = data.get("response", "").strip()
    except (json.JSONDecodeError, KeyError) as exc:
        raise RuntimeError(f"Unexpected Ollama response: {exc}") from exc

    if not answer:
        raise RuntimeError("Ollama returned an empty response.")

    elapsed = int((time.perf_counter() - t0) * 1000)
    logger.info("Ollama generated answer in %d ms (%d chars)", elapsed, len(answer))
    return answer


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_answer(
    question: str,
    context_chunks: list[dict],
    config: Optional[dict] = None,
) -> str:
    """
    Generate a grounded medical answer.

    Provider is selected from config.yaml → llm.provider:
      "gemini"  → Google Gemini API (default)
      "ollama"  → Local Ollama

    Args:
        question       : User's medical question.
        context_chunks : Retrieved context chunks (dicts with 'text' key).
        config         : Config dict (loaded from config.yaml if None).

    Returns:
        Generated answer string.

    Raises:
        RuntimeError   : If the provider is unreachable or returns an error.
    """
    if config is None:
        config = _load_config()

    provider = config.get("llm", {}).get("provider", "gemini").lower()
    prompt = _build_prompt(question, context_chunks)

    if provider == "gemini":
        return _generate_gemini(prompt, config)
    elif provider == "ollama":
        return _generate_ollama(prompt, config)
    else:
        raise RuntimeError(
            f"Unknown LLM provider '{provider}'. "
            "Set llm.provider to 'gemini' or 'ollama' in config.yaml."
        )
