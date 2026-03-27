"""
src/api/main.py — MediRAG FastAPI Application
=============================================
FR-18: Two endpoints:
  GET  /health   → liveness check + Ollama status
  POST /evaluate → calls run_evaluation(), returns FR-17 JSON

Design decisions:
  - DeBERTa model is loaded once at app startup (not per-request)
  - If any module raises an exception, partial results are returned (no HTTP 500)
  - HTTP 422 Pydantic validation errors are automatic
  - RAGAS is disabled by default (run_ragas=False) — set to True only if
    Ollama/OpenAI is available; the RAGAS module already fails gracefully.

To run:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import requests
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    ContextChunk,
    EvaluateRequest,
    EvaluateResponse,
    HealthResponse,
    ModuleResults,
    ModuleScore,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
)
from src.evaluate import run_evaluation
from src.pipeline.generator import generate_answer
from src.pipeline.retriever import Retriever

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
try:
    _cfg = yaml.safe_load(Path("config.yaml").read_text())
    _log_level = _cfg.get("logging", {}).get("level", "INFO")
    _ollama_base = _cfg.get("llm", {}).get("base_url", "http://localhost:11434")
    _api_cfg = _cfg.get("api", {})
except Exception:
    _log_level = "INFO"
    _ollama_base = "http://localhost:11434"
    _api_cfg = {}

logging.basicConfig(
    level=getattr(logging, _log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: warm DeBERTa once at startup so the first request isn't slow
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm DeBERTa and Retriever at startup."""
    logger.info("MediRAG API starting — pre-warming models...")
    try:
        from src.modules.faithfulness import _get_model
        _get_model()
        logger.info("DeBERTa pre-warm complete.")
    except Exception as exc:
        logger.warning("DeBERTa pre-warm skipped: %s", exc)

    # Pre-load the retriever (BioBERT + FAISS index) into app state
    try:
        app.state.retriever = Retriever(_cfg)
        # Trigger lazy load now so first /query request isn't slow
        app.state.retriever._load_model()
        app.state.retriever._load_index()
        logger.info("Retriever pre-warm complete.")
    except Exception as exc:
        logger.warning("Retriever pre-warm skipped: %s", exc)
        app.state.retriever = None

    yield
    logger.info("MediRAG API shutting down.")



# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="MediRAG Evaluation API",
    description=(
        "Evaluate LLM-generated medical answers against retrieved evidence. "
        "Returns faithfulness, entity accuracy, source credibility, "
        "contradiction risk, and a composite Health Risk Score (HRS)."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Allow all origins for local dev / Streamlit on same machine
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper: check Ollama
# ---------------------------------------------------------------------------
def _check_ollama() -> bool:
    """Return True if Ollama API is reachable."""
    try:
        resp = requests.get(f"{_ollama_base}/api/tags", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Helper: convert EvalResult details → ModuleScore
# ---------------------------------------------------------------------------
def _module_score(module_results: dict, key: str) -> Optional[ModuleScore]:
    data = module_results.get(key)
    if data is None:
        return None
    return ModuleScore(
        score=data.get("score", 0.0),
        details=data.get("details", {}),
        error=data.get("error"),
        latency_ms=data.get("latency_ms"),
    )


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """
    Liveness check.

    Returns {"status": "ok", "ollama_available": true/false}.
    Always returns 200 — the caller decides what to do with `ollama_available`.
    """
    return HealthResponse(
        status="ok",
        ollama_available=_check_ollama(),
    )


# ---------------------------------------------------------------------------
# POST /evaluate
# ---------------------------------------------------------------------------
@app.post("/evaluate", response_model=EvaluateResponse, tags=["evaluation"])
def evaluate(req: EvaluateRequest) -> EvaluateResponse:
    """
    Run the full MediRAG evaluation pipeline on a question + answer + context.

    - Validates inputs (FR-18: length limits, chunk count)
    - Runs Faithfulness, Entity Verification, Source Credibility, Contradiction
    - Optionally runs RAGAS (set `run_ragas=true` if Ollama/OpenAI is available)
    - Returns composite Health Risk Score (HRS) + per-module breakdown

    **Note on `run_ragas`**: RAGAS requires a running LLM backend (Ollama or
    OpenAI). If unavailable, RAGAS will gracefully return score=0.5 as a
    neutral fallback — it will NOT crash the request.
    """
    logger.info(
        "POST /evaluate — question=%r, chunks=%d, run_ragas=%s",
        req.question[:80],
        len(req.context_chunks),
        req.run_ragas,
    )

    # Convert Pydantic ContextChunk → plain dicts for the pipeline
    context_dicts: list[dict] = [chunk.model_dump(exclude_none=True) for chunk in req.context_chunks]

    t0 = time.perf_counter()
    try:
        result = run_evaluation(
            question=req.question,
            answer=req.answer,
            context_chunks=context_dicts,
            rxnorm_cache_path=req.rxnorm_cache_path,
            run_ragas=req.run_ragas,
        )
    except Exception as exc:
        logger.exception("run_evaluation raised an unhandled exception: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation pipeline error: {type(exc).__name__}: {exc}",
        ) from exc

    total_ms = int((time.perf_counter() - t0) * 1000)

    # Extract composite score + details
    composite = float(result.score)
    hrs = int(round(100 * (1.0 - composite)))
    hrs = max(0, min(100, hrs))

    details = result.details or {}
    confidence_level = details.get("confidence_level", "UNKNOWN")
    risk_band = details.get("risk_band", "UNKNOWN")
    pipeline_ms = details.get("total_pipeline_ms", total_ms)

    # Build per-module scores
    mod_results: dict = details.get("module_results", {})
    module_scores = ModuleResults(
        faithfulness=_module_score(mod_results, "faithfulness"),
        entity_verifier=_module_score(mod_results, "entity_verifier"),
        source_credibility=_module_score(mod_results, "source_credibility"),
        contradiction=_module_score(mod_results, "contradiction"),
        ragas=_module_score(mod_results, "ragas"),
    )

    logger.info(
        "POST /evaluate → HRS=%d (%s) in %d ms",
        hrs, risk_band, pipeline_ms,
    )

    return EvaluateResponse(
        composite_score=composite,
        hrs=hrs,
        confidence_level=confidence_level,
        risk_band=risk_band,
        module_results=module_scores,
        total_pipeline_ms=pipeline_ms,
    )


# ---------------------------------------------------------------------------
# POST /query  — end-to-end: question → retrieve → generate → evaluate
# ---------------------------------------------------------------------------
@app.post("/query", response_model=QueryResponse, tags=["query"])
def query(req: QueryRequest) -> QueryResponse:
    """
    Full end-to-end MediRAG pipeline.

    1. Retrieves top-k context chunks from FAISS (BioBERT)
    2. Generates a grounded answer using Mistral (Ollama)
    3. Evaluates the answer with all 4 modules + aggregator
    4. Returns the answer, retrieved chunks, HRS score, and full breakdown

    **Requires Ollama running locally with Mistral pulled.**
    No fallback — returns 503 if Ollama is unavailable.
    """
    import time as _time
    t_total = _time.perf_counter()

    logger.info("POST /query — question=%r, top_k=%d", req.question[:80], req.top_k)

    # Step 1: Retrieve
    retriever: Optional[Retriever] = getattr(app.state, "retriever", None)
    if retriever is None:
        # Fallback: instantiate now (slower first call)
        try:
            retriever = Retriever(_cfg)
        except Exception as exc:
            raise HTTPException(status_code=503,
                detail=f"Retriever unavailable: {exc}") from exc

    try:
        raw_results = retriever.search(req.question, top_k=req.top_k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503,
            detail=f"FAISS index not found: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500,
            detail=f"Retrieval error: {exc}") from exc

    if not raw_results:
        raise HTTPException(status_code=404,
            detail="No relevant documents found for this question.")

    # Convert retriever output → chunk dicts for generator + evaluate
    context_chunks: list[dict] = []
    retrieved_chunks_out: list[RetrievedChunk] = []
    for chunk_text, meta, score in raw_results:
        d = {
            "text":       chunk_text,
            "chunk_id":   meta.get("chunk_id"),
            "source":     meta.get("source", ""),
            "pub_type":   meta.get("pub_type", ""),
            "pub_year":   meta.get("pub_year"),
            "title":      meta.get("title", ""),
        }
        context_chunks.append(d)
        retrieved_chunks_out.append(RetrievedChunk(
            chunk_id=meta.get("chunk_id"),
            text=chunk_text[:500],   # truncate for response readability
            source=meta.get("source", ""),
            pub_type=meta.get("pub_type", ""),
            pub_year=meta.get("pub_year"),
            title=meta.get("title", ""),
            similarity_score=round(score, 4),
        ))

    logger.info("Retrieved %d chunks (top score=%.4f)", len(context_chunks),
                raw_results[0][2] if raw_results else 0.0)

    # Convert request overrides into a dict for generator
    llm_overrides = {}
    if req.llm_provider:
        llm_overrides["provider"] = req.llm_provider
    if req.llm_api_key:
        llm_overrides["api_key"] = req.llm_api_key
    if req.llm_model:
        llm_overrides["model"] = req.llm_model
    if req.ollama_url:
        llm_overrides["ollama_url"] = req.ollama_url

    # Step 2: Generate answer via LLM (Gemini or Ollama)
    try:
        answer = generate_answer(req.question, context_chunks, _cfg, overrides=llm_overrides)
    except RuntimeError as exc:
        raise HTTPException(status_code=503,
            detail=f"LLM generation failed: {exc}") from exc

    # Step 3: Evaluate
    try:
        eval_result = run_evaluation(
            question=req.question,
            answer=answer,
            context_chunks=context_chunks,
            run_ragas=req.run_ragas,
        )
    except Exception as exc:
        logger.exception("Evaluation failed: %s", exc)
        raise HTTPException(status_code=500,
            detail=f"Evaluation error: {exc}") from exc

    # Step 4: Build response
    details = eval_result.details or {}
    composite = float(eval_result.score)
    hrs = int(round(100 * (1.0 - composite)))
    hrs = max(0, min(100, hrs))
    mod_results: dict = details.get("module_results", {})

    total_ms = int((_time.perf_counter() - t_total) * 1000)
    logger.info("POST /query → HRS=%d (%s) in %d ms total",
                hrs, details.get("risk_band", "?"), total_ms)

    return QueryResponse(
        question=req.question,
        generated_answer=answer,
        retrieved_chunks=retrieved_chunks_out,
        composite_score=composite,
        hrs=hrs,
        confidence_level=details.get("confidence_level", "UNKNOWN"),
        risk_band=details.get("risk_band", "UNKNOWN"),
        module_results=ModuleResults(
            faithfulness=_module_score(mod_results, "faithfulness"),
            entity_verifier=_module_score(mod_results, "entity_verifier"),
            source_credibility=_module_score(mod_results, "source_credibility"),
            contradiction=_module_score(mod_results, "contradiction"),
            ragas=_module_score(mod_results, "ragas"),
        ),
        total_pipeline_ms=total_ms,
    )
