"""
src/api/schemas.py — Pydantic request/response models for MediRAG FastAPI
=========================================================================
FR-18: Input validation limits from config.yaml → api:
  - max_query_length:  500
  - max_answer_length: 2000
  - max_chunks:        10
  - max_chunk_length:  2000
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class ContextChunk(BaseModel):
    """A single retrieved context chunk passed to the evaluation pipeline."""
    text: str = Field(..., min_length=1, max_length=2000,
                      description="Chunk text (max 2000 chars)")
    # Optional metadata fields — all pass-through to the pipeline modules
    chunk_id: Optional[str] = None
    pub_type: Optional[str] = None
    pub_year: Optional[int] = None
    source: Optional[str] = None
    title: Optional[str] = None
    tier_type: Optional[str] = None       # pre-labelled evidence tier (optional)
    score: Optional[float] = None         # retrieval similarity score


class EvaluateRequest(BaseModel):
    """POST /evaluate — request body."""
    question: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="User question (5–500 chars)",
        examples=["What is the recommended dosage of Metformin for Type 2 Diabetes in elderly patients?"],
    )
    answer: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="LLM-generated answer to evaluate (1–2000 chars)",
        examples=["Metformin is typically started at 500 mg twice daily with meals..."],
    )
    context_chunks: List[ContextChunk] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Retrieved context chunks (1–10 items)",
    )
    run_ragas: bool = Field(
        default=False,
        description="Run RAGAS evaluation (requires Ollama or OpenAI backend; slower)",
    )
    rxnorm_cache_path: str = Field(
        default="data/rxnorm_cache.csv",
        description="Path to RxNorm cache CSV",
    )

    @field_validator("context_chunks")
    @classmethod
    def at_least_one_chunk(cls, v: list) -> list:
        if len(v) == 0:
            raise ValueError("At least one context chunk is required")
        return v


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class ModuleScore(BaseModel):
    """Score + details dict for a single evaluation module."""
    score: float = Field(..., ge=0.0, le=1.0, description="Module score in [0, 1]")
    details: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = Field(None, description="Error message if module failed")
    latency_ms: Optional[int] = None


class ModuleResults(BaseModel):
    """All per-module scores bundled together."""
    faithfulness: Optional[ModuleScore] = None
    entity_verifier: Optional[ModuleScore] = None
    source_credibility: Optional[ModuleScore] = None
    contradiction: Optional[ModuleScore] = None
    ragas: Optional[ModuleScore] = None


class EvaluateResponse(BaseModel):
    """POST /evaluate — response body (FR-17 format)."""
    composite_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Weighted composite score in [0, 1]"
    )
    hrs: int = Field(
        ..., ge=0, le=100,
        description="Health Risk Score = round(100 × (1 - composite_score))"
    )
    confidence_level: str = Field(
        ...,
        description="HIGH / MODERATE / LOW",
    )
    risk_band: str = Field(
        ...,
        description="LOW / MODERATE / HIGH / CRITICAL",
    )
    module_results: ModuleResults
    total_pipeline_ms: int = Field(..., description="Total wall-clock time in ms")


class HealthResponse(BaseModel):
    """GET /health — liveness and dependency status."""
    status: str = Field(default="ok")
    ollama_available: bool
    version: str = Field(default="0.1.0")


# ---------------------------------------------------------------------------
# End-to-end query schemas (POST /query)
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """POST /query — only a question needed; retrieval + generation happen server-side."""
    question: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Medical question (5–500 chars)",
        examples=["What is the recommended dosage of Metformin for elderly Type 2 Diabetes patients?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of context chunks to retrieve (1–10)",
    )
    run_ragas: bool = Field(
        default=False,
        description="Run RAGAS evaluation (requires Ollama — adds ~10s)",
    )


class RetrievedChunk(BaseModel):
    """A single chunk returned alongside the query response for transparency."""
    chunk_id: Optional[str] = None
    text: str
    source: Optional[str] = None
    pub_type: Optional[str] = None
    pub_year: Optional[int] = None
    title: Optional[str] = None
    similarity_score: Optional[float] = None


class QueryResponse(BaseModel):
    """POST /query — full end-to-end response."""
    question: str
    generated_answer: str
    retrieved_chunks: List[RetrievedChunk]
    # Evaluation fields (same as EvaluateResponse)
    composite_score: float = Field(..., ge=0.0, le=1.0)
    hrs: int = Field(..., ge=0, le=100)
    confidence_level: str
    risk_band: str
    module_results: ModuleResults
    total_pipeline_ms: int

