"""
FR-19: src/evaluation/aggregator.py — Weighted Score Aggregation
================================================================
Combines scores from all evaluation modules into a single composite score
using the fixed weights defined in SRS Section 8.2.

Weights (must sum to 1.0):
    faithfulness       : 0.35  (primary signal — DeBERTa NLI)
    entity_accuracy    : 0.20  (SciSpaCy NER + RxNorm)
    source_credibility : 0.20  (evidence tier)
    contradiction_risk : 0.15  (1.0 - contradiction_score)
    ragas_composite    : 0.10  (optional — 0.5 neutral if unavailable)

Output:
    EvalResult with:
        module_name = "aggregator"
        score       = weighted composite in [0, 1]
        details     = {weights_used, weighted_composite, component_contributions}

Usage:
    from src.evaluation.aggregator import aggregate
    agg_result = aggregate(faith_res, entity_res, source_res, contra_res, ragas_res)
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from src.modules.base import EvalResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default weights (SRS Section 8.2)
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: dict[str, float] = {
    "faithfulness":        0.35,
    "entity_accuracy":     0.20,
    "source_credibility":  0.20,
    "contradiction_risk":  0.15,
    "ragas_composite":     0.10,
}


def aggregate(
    faithfulness_result: EvalResult,
    entity_result: EvalResult,
    source_result: EvalResult,
    contradiction_result: EvalResult,
    ragas_result: Optional[EvalResult] = None,
    weights: Optional[dict[str, float]] = None,
) -> EvalResult:
    """
    Aggregate all module scores into a single composite evaluation result.

    Args:
        faithfulness_result    : Output from faithfulness.score_faithfulness()
        entity_result          : Output from entity_verifier.verify_entities()
        source_result          : Output from source_credibility.score_source_credibility()
        contradiction_result   : Output from contradiction.score_contradiction()
        ragas_result           : Output from ragas_eval.score_ragas() (optional)
        weights                : Override default weights (must sum to 1.0)

    Returns:
        EvalResult with module_name="aggregator" and composite score.
    """
    t0 = time.perf_counter()
    w = weights or DEFAULT_WEIGHTS

    # Validate weights sum to 1.0 (tolerance 0.01)
    weight_sum = sum(w.values())
    if abs(weight_sum - 1.0) > 0.01:
        logger.warning(
            "Weights sum to %.4f (expected 1.0) — normalising.", weight_sum
        )
        w = {k: v / weight_sum for k, v in w.items()}

    # Extract scores — use 0.5 neutral for any unavailable module
    faith_score = faithfulness_result.score if not faithfulness_result.error else 0.5
    entity_score = entity_result.score if not entity_result.error else 0.5
    source_score = source_result.score if not source_result.error else 0.5
    contra_score = contradiction_result.score if not contradiction_result.error else 1.0
    ragas_score = (ragas_result.score if ragas_result and not ragas_result.error else 0.5)

    # Compute base weighted contributions
    contributions = {
        "faithfulness_contribution":   round(faith_score  * w["faithfulness"], 4),
        "entity_contribution":         round(entity_score * w["entity_accuracy"], 4),
        "source_contribution":         round(source_score * w["source_credibility"], 4),
        "contradiction_contribution":  round(contra_score * w["contradiction_risk"], 4),
        "ragas_contribution":          round(ragas_score  * w["ragas_composite"], 4),
    }

    base_composite = sum(contributions.values())

    # --- Non-linear Safety Penalties ---
    # If a response has terrible faithfulness (<= 0.6) or high contradiction (<= 0.6),
    # linear weighting isn't enough. We slash the final composite to ensure HRS spikes.
    penalty_multiplier = 1.0
    if faith_score <= 0.6:
        penalty_multiplier *= 0.6  # 40% penalty for making things up
    if contra_score <= 0.6:
        penalty_multiplier *= 0.6  # 40% penalty for contradicting context

    composite = base_composite * penalty_multiplier

    # HRS = round(100 × (1 - composite)), then map to risk band
    hrs = int(round(100 * (1.0 - composite)))
    hrs = max(0, min(100, hrs))

    if hrs <= 30:
        risk_band = "LOW"
    elif hrs <= 60:
        risk_band = "MODERATE"
    elif hrs <= 85:
        risk_band = "HIGH"
    else:
        risk_band = "CRITICAL"

    # Confidence level (based on composite, not HRS)
    if composite >= 0.80:
        confidence = "HIGH"
    elif composite >= 0.55:
        confidence = "MODERATE"
    else:
        confidence = "LOW"

    details = {
        "weights_used": {k: round(v, 4) for k, v in w.items()},
        "component_scores": {
            "faithfulness":       round(faith_score, 4),
            "entity_accuracy":    round(entity_score, 4),
            "source_credibility": round(source_score, 4),
            "contradiction_risk": round(contra_score, 4),
            "ragas_composite":    round(ragas_score, 4),
        },
        "weighted_composite": round(composite, 4),
        "hrs": hrs,
        "risk_band": risk_band,
        "component_contributions": contributions,
        "confidence_level": confidence,
        "module_latencies_ms": {
            "faithfulness":       faithfulness_result.latency_ms,
            "entity_verifier":    entity_result.latency_ms,
            "source_credibility": source_result.latency_ms,
            "contradiction":      contradiction_result.latency_ms,
            "ragas":              ragas_result.latency_ms if ragas_result else 0,
        },
    }

    latency_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(
        "Aggregated score: %.3f (%s confidence) — "
        "faith=%.2f entity=%.2f source=%.2f contra=%.2f ragas=%.2f",
        composite, confidence,
        faith_score, entity_score, source_score, contra_score, ragas_score,
    )

    return EvalResult(
        module_name="aggregator",
        score=composite,
        details=details,
        latency_ms=latency_ms,
    )
