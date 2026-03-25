"""
FR-14: src/modules/source_credibility.py — Module 3: Source Credibility Scoring
=================================================================================
Scores the credibility of retrieved source documents based on their publication
type / evidence tier.

Tier weights (SRS Section 6.3):
    clinical_guideline   → 1.00  (Tier 1 — highest authority)
    systematic_review    → 0.85  (Tier 2)
    research_abstract    → 0.70  (Tier 3 — PubMedQA default)
    review_article       → 0.60  (Tier 4)
    clinical_case        → 0.50  (Tier 5)
    unknown / other      → 0.30  (fallback)

Detection:
    1. Use 'tier_type' metadata field if present (set by embedder.py)
    2. Fall back to keyword matching in pub_type / title text

Score = weighted mean of tier weights across all retrieved chunks.

Each chunk must be a dict with at minimum:
    {"text": str, "metadata": {"tier_type": str, "pub_type": str, "title": str}}
or the simpler form accepted by the retriever:
    {"text": str, "source": str, "tier_type": str, "title": str}
"""
from __future__ import annotations

import logging
import re
import time

from src.modules.base import EvalResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Evidence tier weights
# ---------------------------------------------------------------------------

TIER_WEIGHTS: dict[str, float] = {
    "clinical_guideline":  1.00,
    "systematic_review":   0.85,
    "research_abstract":   0.70,
    "review_article":      0.60,
    "clinical_case":       0.50,
    "unknown":             0.30,
}

# Keyword → tier_type mapping for fallback text matching
_KEYWORD_MAP: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(guideline|clinical practice|recommendation|consensus)\b", re.I), "clinical_guideline"),
    (re.compile(r"\b(systematic review|meta.?analysis)\b", re.I),                    "systematic_review"),
    # RCT / controlled trial → highest single-study evidence tier
    (re.compile(r"\b(randomized|randomised|controlled trial|rct|clinical trial)\b", re.I), "clinical_guideline"),
    (re.compile(r"\b(review|overview)\b", re.I),                                     "review_article"),
    (re.compile(r"\b(case report|case study|clinical case)\b", re.I),                "clinical_case"),
    (re.compile(r"\b(abstract|research article|original article|journal)\b", re.I),  "research_abstract"),
]


def _classify_tier(chunk: dict) -> tuple[str, str | None]:
    """
    Return (tier_type, matched_keyword) for a single retrieved chunk dict.

    Priority 1: explicit tier_type field (set by embedder.py)
    Priority 2: pub_type field directly maps to a known tier name
    Priority 3: keyword regex on pub_type + title text
    """
    # Priority 1: explicit tier_type already set (e.g., by embedder.py)
    tier = (
        chunk.get("tier_type")
        or chunk.get("metadata", {}).get("tier_type")
    )
    if tier and tier in TIER_WEIGHTS:
        return tier, None

    # Priority 2: direct pub_type value lookup
    # Handles underscore-separated values like "research_abstract" which
    # won't match word-boundary regex patterns
    pub_type_raw = str(
        chunk.get("pub_type") or chunk.get("metadata", {}).get("pub_type") or ""
    ).strip().lower()

    _PUB_TYPE_DIRECT: dict[str, str] = {
        "research_abstract":  "research_abstract",
        "abstract":           "research_abstract",
        "systematic_review":  "systematic_review",
        "systematic review":  "systematic_review",
        "meta_analysis":      "systematic_review",
        "meta-analysis":      "systematic_review",
        "clinical_guideline": "clinical_guideline",
        "clinical guideline": "clinical_guideline",
        "guideline":          "clinical_guideline",
        "review_article":     "review_article",
        "review article":     "review_article",
        "review":             "review_article",
        "clinical_case":      "clinical_case",
        "case_report":        "clinical_case",
        "case report":        "clinical_case",
    }
    if pub_type_raw in _PUB_TYPE_DIRECT:
        return _PUB_TYPE_DIRECT[pub_type_raw], None

    # Priority 3: keyword regex on pub_type + title text
    title = str(chunk.get("title") or chunk.get("metadata", {}).get("title") or "")
    text_to_search = f"{pub_type_raw} {title}"

    for pattern, matched_tier in _KEYWORD_MAP:
        m = pattern.search(text_to_search)
        if m:
            return matched_tier, m.group(0)


    return "unknown", None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_source_credibility(
    retrieved_chunks: list[dict],
) -> EvalResult:
    """
    Score the credibility of a set of retrieved source documents.

    Args:
        retrieved_chunks : List of chunk dicts as returned by retriever.retrieve().
                          Each must contain at minimum 'text' and ideally
                          'tier_type', 'pub_type', 'title', 'chunk_id' fields.

    Returns:
        EvalResult with module_name="source_credibility", score in [0,1], and
        details matching the shape from src/modules/__init__.py.
    """
    t0 = time.perf_counter()

    if not retrieved_chunks:
        return EvalResult(
            module_name="source_credibility",
            score=0.0,
            details={"chunks": [], "method_used": "none"},
            error="No chunks provided",
            latency_ms=0,
        )

    chunk_details: list[dict] = []
    weights: list[float] = []
    method_used = "metadata"  # assume metadata-first; may switch to keyword

    for i, chunk in enumerate(retrieved_chunks):
        tier_type, matched_kw = _classify_tier(chunk)
        weight = TIER_WEIGHTS.get(tier_type, TIER_WEIGHTS["unknown"])
        weights.append(weight)

        if matched_kw:
            method_used = "keyword"

        # Compute tier number (1-5) for display
        tier_num = {
            "clinical_guideline": 1,
            "systematic_review":  2,
            "research_abstract":  3,
            "review_article":     4,
            "clinical_case":      5,
        }.get(tier_type, 6)  # 6 = unknown/unclassified

        chunk_details.append(
            {
                "chunk_id": chunk.get("chunk_id") or chunk.get("metadata", {}).get("chunk_id") or f"chunk_{i}",
                "tier": tier_num,
                "tier_type": tier_type,
                "tier_weight": round(weight, 2),
                "pub_type": chunk.get("pub_type") or chunk.get("metadata", {}).get("pub_type") or "",
                "title": (chunk.get("title") or chunk.get("metadata", {}).get("title") or "")[:80],
                "matched_keyword": matched_kw,
            }
        )

    score = sum(weights) / len(weights) if weights else 0.0

    details = {
        "method_used": method_used,
        "chunk_count": len(retrieved_chunks),
        "avg_tier_weight": round(score, 4),
        "chunks": chunk_details,
    }

    latency_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(
        "Source credibility: %.3f (avg tier weight over %d chunks) in %d ms",
        score, len(retrieved_chunks), latency_ms,
    )
    return EvalResult(
        module_name="source_credibility",
        score=score,
        details=details,
        latency_ms=latency_ms,
    )
