"""
FR-05: src/modules/faithfulness.py — Module 1: Faithfulness Scoring
=====================================================================
Uses cross-encoder/nli-deberta-v3-small to score how well the LLM answer
is entailed by the retrieved context chunks.

Architecture:
    1. Split answer into individual claims (sentences via pysbd)
    2. For each claim: compute NLI score against every context chunk
    3. Assign claim status: ENTAILED / NEUTRAL / CONTRADICTED
    4. score = entailed_count / total_claims

Thresholds (SRS Section 6.1):
    entailment  ≥ 0.50  → ENTAILED
    contradiction ≥ 0.30 → CONTRADICTED
    otherwise           → NEUTRAL

Model loaded lazily and cached at module level (avoids double-loading
when called multiple times in same process).
"""
from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import TYPE_CHECKING

import pysbd
from sentence_transformers import CrossEncoder

from src.modules.base import EvalResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

# NLI label order returned by cross-encoder/nli-deberta-v3-small:
# index 0 = contradiction, index 1 = neutral, index 2 = entailment
LABEL_CONTRADICTION = 0
LABEL_NEUTRAL = 1
LABEL_ENTAILMENT = 2

ENTAILMENT_THRESHOLD = 0.50
CONTRADICTION_THRESHOLD = 0.30

# ---------------------------------------------------------------------------
# Lazy model loader
# ---------------------------------------------------------------------------

_model: CrossEncoder | None = None
_segmenter = pysbd.Segmenter(language="en", clean=False)


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        logger.info("Loading NLI model: %s (first call only)", MODEL_NAME)
        _model = CrossEncoder(MODEL_NAME)
        logger.info("NLI model loaded.")
    return _model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_faithfulness(
    answer: str,
    context_docs: list[str],
    chunk_ids: list[str] | None = None,
    max_chunks: int = 5,
) -> EvalResult:
    """
    Score the faithfulness of an answer against retrieved context documents.

    Args:
        answer       : The LLM-generated answer text.
        context_docs : List of context passage strings (top-k retrieved chunks).
        chunk_ids    : Optional IDs matching context_docs for traceability.
        max_chunks   : Maximum context chunks to consider (to limit API calls).

    Returns:
        EvalResult with module_name="faithfulness", score in [0,1], and details
        dict matching the shape defined in src/modules/__init__.py.
    """
    t0 = time.perf_counter()

    if not answer or not context_docs:
        return EvalResult(
            module_name="faithfulness",
            score=0.0,
            details={"error": "Empty answer or no context provided"},
            error="Empty answer or no context",
            latency_ms=0,
        )

    # Limit context size
    docs = context_docs[:max_chunks]
    ids = (chunk_ids or [f"chunk_{i}" for i in range(len(docs))])[:max_chunks]

    # Split answer into claims
    try:
        claims: list[str] = [s.strip() for s in _segmenter.segment(answer) if s.strip()]
    except Exception:
        claims = [s.strip() for s in answer.split(".") if s.strip()]

    if not claims:
        return EvalResult(
            module_name="faithfulness",
            score=0.0,
            details={"error": "Could not extract claims from answer"},
            error="No claims extracted",
            latency_ms=0,
        )

    model = _get_model()

    # Build (claim, context) pairs for batch inference
    all_pairs = []
    pair_map: list[tuple[int, int]] = []  # (claim_idx, doc_idx)
    for ci, claim in enumerate(claims):
        for di, doc in enumerate(docs):
            all_pairs.append((claim, doc))
            pair_map.append((ci, di))

    # Batch NLI inference
    try:
        scores_raw = model.predict(all_pairs, apply_softmax=True)
    except Exception as exc:
        logger.error("NLI model inference failed: %s", exc)
        return EvalResult(
            module_name="faithfulness",
            score=0.0,
            details={},
            error=f"Model inference error: {exc}",
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )

    # Aggregate: for each claim find the context with the highest entailment
    claim_results: list[dict] = []
    entailed = 0
    neutral = 0
    contradicted = 0

    for ci, claim in enumerate(claims):
        # Collect all (doc_idx, score_vec) for this claim
        best_entailment = 0.0
        best_contradiction = 0.0
        best_doc_idx = 0

        for idx, (c_i, d_i) in enumerate(pair_map):
            if c_i != ci:
                continue
            score_vec = scores_raw[idx]
            ent_score = float(score_vec[LABEL_ENTAILMENT])
            con_score = float(score_vec[LABEL_CONTRADICTION])
            if ent_score > best_entailment:
                best_entailment = ent_score
                best_doc_idx = d_i
                best_contradiction = con_score

        # Classify
        if best_entailment >= ENTAILMENT_THRESHOLD:
            status = "ENTAILED"
            nli_score = best_entailment
            entailed += 1
        elif best_contradiction >= CONTRADICTION_THRESHOLD:
            status = "CONTRADICTED"
            nli_score = best_contradiction
            contradicted += 1
        else:
            status = "NEUTRAL"
            nli_score = best_entailment
            neutral += 1

        claim_results.append(
            {
                "claim": claim,
                "status": status,
                "best_chunk_id": ids[best_doc_idx],
                "nli_score": round(nli_score, 4),
            }
        )

    total = len(claims)
    score = entailed / total if total > 0 else 0.0

    details = {
        "total_claims": total,
        "entailed_count": entailed,
        "neutral_count": neutral,
        "contradicted_count": contradicted,
        "claims": claim_results,
    }

    latency_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(
        "Faithfulness: %.3f (%d/%d entailed) in %d ms",
        score, entailed, total, latency_ms,
    )
    return EvalResult(
        module_name="faithfulness",
        score=score,
        details=details,
        latency_ms=latency_ms,
    )
