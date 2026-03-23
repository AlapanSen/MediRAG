"""
src/modules/base.py — Shared EvalResult dataclass.
Used as the standard output schema by all 4 evaluation modules.
Details shape per module is fully specified here (SRS Section 5).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """
    Shared output schema for all evaluation modules.

    Attributes:
        module_name : Identifier string, e.g. "faithfulness"
        score       : Module score in [0.0, 1.0] — clipped automatically
        details     : Module-specific dict (see DETAILS SHAPES below)
        error       : None if successful; error message string if module failed
        latency_ms  : Wall-clock milliseconds for this module's execution
    """

    module_name: str
    score: float
    details: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    latency_ms: int = 0

    def __post_init__(self) -> None:
        """Clip score to [0.0, 1.0] as required by SRS 4.2."""
        if not (0.0 <= self.score <= 1.0):
            logger.warning(
                "%s: score %.4f out of [0,1], clipping.",
                self.module_name,
                self.score,
            )
            self.score = max(0.0, min(1.0, self.score))

    # -------------------------------------------------------------------------
    # DETAILS SHAPE REFERENCE (SRS Section 5)
    # -------------------------------------------------------------------------
    #
    # faithfulness.details:
    # {
    #   "total_claims": int,
    #   "entailed_count": int,
    #   "neutral_count": int,
    #   "contradicted_count": int,
    #   "claims": [
    #     {
    #       "claim": str,
    #       "status": "ENTAILED" | "NEUTRAL" | "CONTRADICTED",
    #       "best_chunk_id": str,      # chunk with highest NLI score
    #       "nli_score": float
    #     }
    #   ]
    # }
    #
    # entity_verifier.details:
    # {
    #   "total_entities": int,
    #   "verified_count": int,
    #   "flagged_count": int,
    #   "entities": [
    #     {
    #       "entity": str,
    #       "type": "DRUG" | "DOSAGE" | "CONDITION" | "PROCEDURE",
    #       "status": "VERIFIED" | "FLAGGED" | "NOT_FOUND",
    #       "severity": "CRITICAL" | "MODERATE" | "MINOR" | null,
    #       "answer_value": str,
    #       "context_value": str | null,
    #       "rxcui": str | null
    #     }
    #   ]
    # }
    #
    # source_credibility.details:
    # {
    #   "method_used": "keyword" | "metadata",
    #   "chunks": [
    #     {
    #       "chunk_id": str,
    #       "tier": int,             # 1–5
    #       "tier_weight": float,
    #       "pub_type": str,
    #       "title": str,
    #       "matched_keyword": str | null
    #     }
    #   ]
    # }
    #
    # contradiction.details:
    # {
    #   "total_sentences": int,
    #   "checked_pairs": int,
    #   "contradicted_pairs": int,
    #   "pairs": [
    #     {
    #       "sentence_a": str,
    #       "sentence_b": str,
    #       "contradiction_score": float,
    #       "flagged": bool
    #     }
    #   ]
    # }
    #
    # aggregator.details:
    # {
    #   "weights_used": {
    #     "faithfulness": float,
    #     "entity_accuracy": float,
    #     "source_credibility": float,
    #     "contradiction_risk": float
    #   },
    #   "weighted_composite": float,
    #   "component_contributions": {
    #     "faithfulness_contribution": float,
    #     "entity_contribution": float,
    #     "source_contribution": float,
    #     "contradiction_contribution": float
    #   }
    # }
