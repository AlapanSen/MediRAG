"""
FR-06: src/evaluation/ragas_eval.py — RAGAS Faithfulness + Answer Relevancy
=============================================================================
Wraps the ragas library to compute:
    - faithfulness      : context-grounded claim verification
    - answer_relevancy  : semantic similarity of answer to question

Requires an LLM backend. Supported backends (in priority order):
    1. Ollama (local, free)  — set OLLAMA_HOST env var or use default localhost:11434
    2. OpenAI API            — set OPENAI_API_KEY env var
    3. Graceful degradation  — returns score=None with explanation if no LLM available

Usage:
    from src.evaluation.ragas_eval import score_ragas
    result = score_ragas(question, answer, context_docs)

SRS reference: FR-06, Section 7 (Evaluation Pipeline)
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

from src.modules.base import EvalResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _detect_llm_backend() -> Optional[str]:
    """Return 'ollama', 'openai', or None."""
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    # Check if Ollama is running locally
    try:
        import requests
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        resp = requests.get(f"{host}/api/tags", timeout=2)
        if resp.status_code == 200:
            return "ollama"
    except Exception:
        pass
    return None


def _build_ragas_llm(backend: str):
    """Build a ragas-compatible LLM wrapper."""
    if backend == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    elif backend == "ollama":
        from langchain_community.chat_models import ChatOllama
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "mistral")
        return ChatOllama(base_url=host, model=model)
    raise ValueError(f"Unknown backend: {backend}")


def _build_ragas_embeddings(backend: str):
    """Build a ragas-compatible embeddings wrapper."""
    if backend == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings()
    elif backend == "ollama":
        from langchain_community.embeddings import OllamaEmbeddings
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        return OllamaEmbeddings(base_url=host, model=model)
    raise ValueError(f"Unknown backend: {backend}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_ragas(
    question: str,
    answer: str,
    context_docs: list[str],
    max_contexts: int = 3,
) -> EvalResult:
    """
    Compute RAGAS faithfulness and answer_relevancy scores.

    Args:
        question     : Original user question.
        answer       : LLM-generated answer.
        context_docs : Retrieved context passages.
        max_contexts : Max context chunks to pass to RAGAS (to limit token cost).

    Returns:
        EvalResult with module_name="ragas", score in [0,1].
        score = mean(faithfulness, answer_relevancy).
        Returns score=0.5 (neutral) if no LLM backend is available.
    """
    t0 = time.perf_counter()

    backend = _detect_llm_backend()
    if backend is None:
        logger.warning(
            "No LLM backend available for RAGAS. "
            "Set OPENAI_API_KEY or start Ollama (ollama serve). "
            "Returning neutral score (0.5)."
        )
        return EvalResult(
            module_name="ragas",
            score=0.5,
            details={
                "backend": None,
                "faithfulness": None,
                "answer_relevancy": None,
                "note": "No LLM backend — set OPENAI_API_KEY or start Ollama",
            },
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy

        llm = _build_ragas_llm(backend)
        embeddings = _build_ragas_embeddings(backend)

        # Configure metrics to use our chosen backend
        faithfulness.llm = llm
        faithfulness.embeddings = embeddings
        answer_relevancy.llm = llm
        answer_relevancy.embeddings = embeddings

        contexts = context_docs[:max_contexts]
        dataset = Dataset.from_dict(
            {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
            }
        )

        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])

        faith_score = float(result["faithfulness"])
        relevancy_score = float(result["answer_relevancy"])
        composite = (faith_score + relevancy_score) / 2.0

        details = {
            "backend": backend,
            "faithfulness": round(faith_score, 4),
            "answer_relevancy": round(relevancy_score, 4),
        }

        latency_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "RAGAS: faith=%.3f, relevancy=%.3f → composite=%.3f in %d ms",
            faith_score, relevancy_score, composite, latency_ms,
        )
        return EvalResult(
            module_name="ragas",
            score=composite,
            details=details,
            latency_ms=latency_ms,
        )

    except Exception as exc:
        logger.error("RAGAS evaluation failed: %s", exc)
        return EvalResult(
            module_name="ragas",
            score=0.5,
            details={"backend": backend, "error": str(exc)},
            error=str(exc),
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )
