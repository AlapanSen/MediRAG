"""
FR-17: src/modules/contradiction.py — Module 4: Cross-Document Contradiction Detection
========================================================================================
Uses the same DeBERTa NLI cross-encoder (cross-encoder/nli-deberta-v3-small) to
detect contradictions between the LLM answer and retrieved context passages.

Algorithm (SRS Section 6.4):
    1. Split answer into sentences  (claims)
    2. Split each context chunk into sentences
    3. For each (answer_sentence, context_sentence) pair:
        - Run NLI → get contradiction score
        - If contradiction_score ≥ CONTRADICTION_THRESHOLD → flag pair
    4. score = 1.0 - (flagged_pairs / total_pairs)

This module shares the NLI model instance with faithfulness.py when both
run in the same process (the model is cached at the faithfulness module level).

Design note:
    To keep latency manageable, context sentences are limited to
    MAX_CONTEXT_SENTS per chunk and total pairs are capped at MAX_PAIRS.
"""
from __future__ import annotations

import logging
import time

import pysbd

from src.modules.base import EvalResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTRADICTION_THRESHOLD = 0.75   # Conservative: only flag genuine medical contradictions
MIN_KEYWORD_OVERLAP = 1          # At least 1 meaningful word in common before running NLI
MAX_CONTEXT_SENTS = 4   # top N sentences per context chunk
MAX_PAIRS = 200          # hard cap to keep latency bounded (~2-3s)

_segmenter = pysbd.Segmenter(language="en", clean=False)

# Common stopwords to ignore in overlap check
_STOPWORDS = {
    "the", "a", "an", "is", "in", "of", "to", "for", "and", "or", "are",
    "be", "at", "by", "if", "it", "as", "on", "with", "this", "that",
    "was", "were", "not", "no", "have", "has", "had", "but", "so", "from",
    "should", "may", "can", "will", "than", "more", "when", "which", "who",
    "what", "all", "each", "after", "before", "been", "do", "does", "1",
    "2", "3", "mg", "iv", "od", "per", "day", "based", "using", "include",
}


def _keyword_overlap(sent_a: str, sent_b: str) -> int:
    """Count shared content words between two sentences."""
    tokens_a = {w.lower() for w in sent_a.split() if w.lower() not in _STOPWORDS and len(w) > 2}
    tokens_b = {w.lower() for w in sent_b.split() if w.lower() not in _STOPWORDS and len(w) > 2}
    return len(tokens_a & tokens_b)


def _segment(text: str) -> list[str]:
    try:
        return [s.strip() for s in _segmenter.segment(text) if s.strip()]
    except Exception:
        return [s.strip() for s in text.split(".") if s.strip()]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_contradiction(
    answer: str,
    context_docs: list[str],
    max_chunks: int = 5,
) -> EvalResult:
    """
    Detect contradictions between the LLM answer and retrieved context.

    Args:
        answer       : LLM-generated answer text.
        context_docs : List of retrieved context passage strings.
        max_chunks   : Max number of context chunks to evaluate.

    Returns:
        EvalResult with module_name="contradiction", score in [0,1] where
        1.0 = no contradictions detected, 0.0 = all pairs contradicted.
    """
    t0 = time.perf_counter()

    if not answer or not context_docs:
        return EvalResult(
            module_name="contradiction",
            score=1.0,  # neutral — nothing to contradict
            details={"total_sentences": 0, "checked_pairs": 0, "contradicted_pairs": 0, "pairs": []},
            latency_ms=0,
        )

    # Import model via faithfulness module (shared cache)
    try:
        from src.modules.faithfulness import _get_model, LABEL_CONTRADICTION
    except ImportError:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder("cross-encoder/nli-deberta-v3-small")
        _get_model = lambda: _model  # noqa: E731
        LABEL_CONTRADICTION = 0

    model = _get_model()

    # Segment answer into claims
    answer_sents = _segment(answer)
    if not answer_sents:
        return EvalResult(
            module_name="contradiction",
            score=1.0,
            details={"total_sentences": 0, "checked_pairs": 0, "contradicted_pairs": 0, "pairs": []},
            latency_ms=0,
        )

    # Segment context chunks
    docs = context_docs[:max_chunks]
    context_sents: list[str] = []
    for doc in docs:
        sents = _segment(doc)[:MAX_CONTEXT_SENTS]
        context_sents.extend(sents)

    if not context_sents:
        return EvalResult(
            module_name="contradiction",
            score=1.0,
            details={"total_sentences": len(answer_sents), "checked_pairs": 0, "contradicted_pairs": 0, "pairs": []},
            latency_ms=0,
        )

    # Build pairs WITH topical pre-filter (skip unrelated sentence pairs entirely)
    all_pairs: list[tuple[str, str]] = []
    for a_sent in answer_sents:
        for c_sent in context_sents:
            if _keyword_overlap(a_sent, c_sent) >= MIN_KEYWORD_OVERLAP:
                all_pairs.append((a_sent, c_sent))
            if len(all_pairs) >= MAX_PAIRS:
                break
        if len(all_pairs) >= MAX_PAIRS:
            break

    if not all_pairs:
        # All sentence pairs were topically unrelated — no contradiction possible
        return EvalResult(
            module_name="contradiction",
            score=1.0,
            details={"total_sentences": len(answer_sents), "checked_pairs": 0, "contradicted_pairs": 0, "pairs": []},
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )

    # Batch NLI inference
    try:
        scores_raw = model.predict(all_pairs, apply_softmax=True)
    except Exception as exc:
        logger.error("Contradiction NLI inference failed: %s", exc)
        return EvalResult(
            module_name="contradiction",
            score=1.0,
            details={},
            error=f"Model inference error: {exc}",
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )

    # Collect flagged pairs
    pair_details: list[dict] = []
    contradicted = 0
    total = len(all_pairs)

    for i, (a_sent, c_sent) in enumerate(all_pairs):
        con_score = float(scores_raw[i][LABEL_CONTRADICTION])
        flagged = con_score >= CONTRADICTION_THRESHOLD
        if flagged:
            contradicted += 1
            # Only log the most severe contradictions to keep details manageable
            pair_details.append(
                {
                    "sentence_a": a_sent[:120],
                    "sentence_b": c_sent[:120],
                    "contradiction_score": round(con_score, 4),
                    "flagged": True,
                }
            )

    # Score: 1.0 = clean, lower = more contradictions found
    score = 1.0 - (contradicted / total) if total > 0 else 1.0

    details = {
        "total_sentences": len(answer_sents),
        "checked_pairs": total,
        "contradicted_pairs": contradicted,
        "pairs": pair_details[:20],  # cap output to top 20 flagged pairs
    }

    latency_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(
        "Contradiction: %.3f (%d/%d pairs flagged) in %d ms",
        score, contradicted, total, latency_ms,
    )
    return EvalResult(
        module_name="contradiction",
        score=score,
        details=details,
        latency_ms=latency_ms,
    )
