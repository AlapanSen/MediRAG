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

import threading
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
    IngestRequest,
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

    # [DEMO MODE] Inject a false claim to demonstrate the intervention system
    if req.inject_hallucination:
        logger.warning("DEMO MODE: Injecting hallucinated claim into answer: '%s'",
                       req.inject_hallucination)
        answer = answer + " " + req.inject_hallucination.strip()

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

    # =========================================================================
    # Step 3b: INTERVENTION LOOP — MediRAG acts on evaluation results
    # =========================================================================
    from src.pipeline.generator import generate_strict_answer

    details = eval_result.details or {}
    composite = float(eval_result.score)
    hrs = int(round(100 * (1.0 - composite)))
    hrs = max(0, min(100, hrs))
    mod_results: dict = details.get("module_results", {})

    intervention_applied = False
    intervention_reason = None
    original_answer = None
    intervention_details = None

    faith_score = (mod_results.get("faithfulness") or {}).get("score", 1.0)

    # Tier 1: CRITICAL BLOCK (HRS ≥ 86) — response is too dangerous to show
    if hrs >= 86:
        original_answer = answer
        answer = (
            "⛔ UNSAFE RESPONSE BLOCKED by MediRAG Safety Gate.\n\n"
            "The generated answer was flagged as CRITICAL risk "
            f"(Health Risk Score: {hrs}/100). "
            "It showed signs of hallucination or contradiction with the retrieved evidence. "
            "Please consult a qualified medical professional or rephrase your question."
        )
        intervention_applied = True
        intervention_reason = "CRITICAL_BLOCKED"
        intervention_details = {
            "hrs_original": hrs,
            "faithfulness": faith_score,
            "message": "Response blocked: HRS ≥ 86 (CRITICAL risk band).",
        }
        logger.warning("INTERVENTION: CRITICAL_BLOCKED — HRS=%d", hrs)

    # Tier 2: HIGH RISK REGENERATION (HRS ≥ 40 or faithfulness < 1.0)
    elif hrs >= 40 or faith_score < 1.0:
        original_answer = answer
        original_hrs = hrs
        logger.warning(
            "INTERVENTION: HIGH_RISK_REGENERATED — HRS=%d, faith=%.2f. Regenerating with strict prompt.",
            hrs, faith_score
        )
        try:
            answer = generate_strict_answer(req.question, context_chunks, _cfg, overrides=llm_overrides)
            # Re-evaluate the corrected answer
            eval_result = run_evaluation(
                question=req.question,
                answer=answer,
                context_chunks=context_chunks,
                run_ragas=False,  # skip RAGAS on retry to reduce latency
            )
            details = eval_result.details or {}
            composite = float(eval_result.score)
            hrs = int(round(100 * (1.0 - composite)))
            hrs = max(0, min(100, hrs))
            mod_results = details.get("module_results", {})
        except Exception as exc:
            logger.error("Strict regeneration failed: %s — keeping original answer", exc)
            answer = original_answer  # fall back gracefully
            original_answer = None

        intervention_applied = True
        intervention_reason = "HIGH_RISK_REGENERATED"
        intervention_details = {
            "hrs_original": original_hrs,
            "hrs_corrected": hrs,
            "faithfulness_original": faith_score,
            "faithfulness_corrected": (mod_results.get("faithfulness") or {}).get("score", 0),
            "message": "Response regenerated with strict context-only prompt due to high risk score.",
        }
    # =========================================================================

    # Step 4: Build response
    total_ms = int((_time.perf_counter() - t_total) * 1000)
    logger.info("POST /query → HRS=%d (%s) intervention=%s in %d ms total",
                hrs, details.get("risk_band", "?"), intervention_reason or "none", total_ms)

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
        intervention_applied=intervention_applied,
        intervention_reason=intervention_reason,
        original_answer=original_answer,
        intervention_details=intervention_details,
    )

# ---------------------------------------------------------------------------
# POST /ingest — dynamically append new documents to the FAISS index
# ---------------------------------------------------------------------------
_faiss_lock = threading.Lock()

@app.post("/ingest", tags=["ingestion"])
def ingest_document(req: IngestRequest):
    """
    Dynamically ingest a new document into the running FAISS index.
    Thread-safe implementation uses a lock to prevent concurrent write corruption.
    """
    import pickle
    import faiss
    from src.pipeline.chunker import chunk_documents


    retriever = getattr(app.state, "retriever", None)
    if retriever is None or retriever._index is None:
        raise HTTPException(status_code=503, detail="Retriever not pre-warmed. Cannot ingest.")

    logger.info("POST /ingest — title='%s', size=%d chars", req.title, len(req.text))
    
    # 1. Chunk the document
    doc = {
        "text": req.text,
        "doc_id": "custom_" + req.title[:10],
        "title": req.title,
        "source": req.source,
        "pub_type": req.pub_type,
        "pub_year": 2026,
    }
    chunks = chunk_documents([doc], _cfg)
    
    if not chunks:
        raise HTTPException(status_code=400, detail="Document produced 0 chunks.")

    # 2. Embed the chunks using the same BioBERT model as the retriever
    from src.pipeline.embedder import encode_texts
    import numpy as np

    # Reuse already-loaded SentenceTransformer from the retriever to avoid double RAM load
    if retriever._model is None:
        retriever._load_model()
    st_model = retriever._model
    
    texts = [c["chunk_text"] for c in chunks]
    embeddings = np.array(st_model.encode(texts, show_progress_bar=False), dtype=np.float32)
    faiss.normalize_L2(embeddings)  # Required: index is IndexFlatIP = cosine sim

    # 3. Thread-safe Index Update with atomic disk writes
    with _faiss_lock:
        import os
        idx_path = Path(_cfg["retrieval"]["index_path"])
        meta_path = Path(_cfg["retrieval"]["metadata_path"])
        
        index = retriever._index
        metadata_store = retriever._metadata

        start_id = len(metadata_store)
        
        # Add to in-memory structures
        for i, chunk in enumerate(chunks):
            metadata_store[start_id + i] = chunk
            
        # Add to FAISS in memory
        index.add(embeddings)
        
        # Atomic FAISS write: write to temp → rename (never leaves a half-written file)
        idx_tmp = str(idx_path) + ".tmp"
        faiss.write_index(index, idx_tmp)
        os.replace(idx_tmp, str(idx_path))
        
        # Atomic metadata write
        meta_tmp = str(meta_path) + ".tmp"
        with open(meta_tmp, "wb") as f:
            pickle.dump(metadata_store, f)
        os.replace(meta_tmp, str(meta_path))
            
    logger.info("Successfully injected %d chunks for '%s' into FAISS.", len(chunks), req.title)
    return {"status": "success", "chunks_added": len(chunks), "title": req.title}
