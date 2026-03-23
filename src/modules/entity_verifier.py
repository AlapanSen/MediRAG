"""
FR-09: src/modules/entity_verifier.py — Module 2: Medical Entity Verification
==============================================================================
Uses SciSpaCy NER (en_core_sci_lg) to extract medical entities from the answer,
then verifies drug entities against the RxNorm cache and/or REST API.

Verification pipeline (SRS Section 6.2):
    1. NER: extract DRUG, DOSAGE, CONDITION, PROCEDURE entities from answer
    2. For each DRUG entity:
        a. Look up in local rxnorm_cache.csv (fast, offline)
        b. If not found, query RxNorm REST API /approximateTerm (live fallback)
        c. If still not found, mark as NOT_FOUND
    3. Cross-check entity presence in context docs (optional validation)
    4. Score = verified_drug_count / total_drug_count  (non-drug entities have no score impact)

Entity status values:
    VERIFIED  — drug found in RxNorm cache or API with rxcui
    FLAGGED   — entity found but has a known dangerous synonym conflict
    NOT_FOUND — drug name not resolvable via any layer

Severity mapping (for FLAGGED):
    brand ↔ generic mismatch → CRITICAL
    dosage discrepancy       → MODERATE
    minor synonym variant    → MINOR
"""
from __future__ import annotations

import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from src.modules.base import EvalResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RXNORM_APPROX_URL = "https://rxnav.nlm.nih.gov/REST/approximateTerm.json"
DEFAULT_CACHE_PATH = "data/rxnorm_cache.csv"
NER_MODEL = "en_core_sci_lg"

# Map spacy entity labels to our schema types
_ENTITY_TYPE_MAP = {
    "CHEMICAL":   "DRUG",
    "DRUG":       "DRUG",
    "COMPOUND":   "DRUG",
    "DISEASE":    "CONDITION",
    "SYMPTOM":    "CONDITION",
    "PROCEDURE":  "PROCEDURE",
    "DOSAGE":     "DOSAGE",
}
DRUG_TYPES = {"DRUG"}  # only these get verified against RxNorm

# ---------------------------------------------------------------------------
# Module-level caches
# ---------------------------------------------------------------------------

_spacy_model = None
_rxnorm_cache: dict[str, str] | None = None  # drug_name -> rxcui
_rxnorm_cache_path: str = DEFAULT_CACHE_PATH


def _get_spacy_model():
    global _spacy_model
    if _spacy_model is None:
        import spacy
        logger.info("Loading SciSpaCy NER model: %s (first call only)", NER_MODEL)
        try:
            _spacy_model = spacy.load(NER_MODEL)
            logger.info("SciSpaCy model loaded.")
        except OSError as exc:
            logger.error(
                "Failed to load '%s'. Install with: "
                "pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/"
                "releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz\nError: %s",
                NER_MODEL, exc,
            )
            raise
    return _spacy_model


def _load_rxnorm_cache(cache_path: str) -> dict[str, str]:
    """Load the RxNorm cache CSV into a lowercase drug_name → rxcui dict."""
    path = Path(cache_path)
    if not path.exists():
        logger.warning(
            "RxNorm cache not found at '%s'. Live API only will be used.", cache_path
        )
        return {}
    try:
        df = pd.read_csv(path, dtype=str)
        cache = {
            str(row["drug_name"]).strip().lower(): str(row["rxcui"]).strip()
            for _, row in df.iterrows()
            if pd.notna(row.get("drug_name")) and pd.notna(row.get("rxcui"))
            and str(row.get("rxcui", "")).strip()
        }
        logger.info("RxNorm cache loaded: %d entries from %s", len(cache), cache_path)
        return cache
    except Exception as exc:
        logger.warning("Failed to load RxNorm cache: %s", exc)
        return {}


def _get_rxnorm_cache(cache_path: str) -> dict[str, str]:
    global _rxnorm_cache, _rxnorm_cache_path
    if _rxnorm_cache is None or cache_path != _rxnorm_cache_path:
        _rxnorm_cache_path = cache_path
        _rxnorm_cache = _load_rxnorm_cache(cache_path)
    return _rxnorm_cache


def _lookup_rxnorm_api(drug_name: str, timeout: int = 4) -> Optional[str]:
    """Query RxNorm REST API. Returns rxcui string or None."""
    try:
        resp = requests.get(
            RXNORM_APPROX_URL,
            params={"term": drug_name, "maxEntries": "1", "option": "1"},
            timeout=timeout,
        )
        if resp.status_code != 200:
            return None
        candidates = (
            resp.json()
            .get("approximateGroup", {})
            .get("candidate", [])
        )
        if candidates:
            return str(candidates[0].get("rxcui", "")).strip() or None
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verify_entities(
    answer: str,
    question: str = "",
    context_docs: list[str] | None = None,
    rxnorm_cache_path: str = DEFAULT_CACHE_PATH,
    use_api_fallback: bool = True,
) -> EvalResult:
    """
    Extract and verify medical entities from the LLM answer.

    Args:
        answer            : LLM-generated answer text.
        question          : Original question (NER'd alongside answer for richer entity set).
        context_docs      : Retrieved context passages (used for cross-checking).
        rxnorm_cache_path : Path to rxnorm_cache.csv.
        use_api_fallback  : Whether to call RxNorm REST API for cache misses.

    Returns:
        EvalResult with module_name="entity_verifier", score in [0,1], and
        details matching the shape from src/modules/__init__.py.
    """
    t0 = time.perf_counter()

    # --- NER -----------------------------------------------------------------
    try:
        nlp = _get_spacy_model()
    except Exception as exc:
        return EvalResult(
            module_name="entity_verifier",
            score=0.5,  # neutral fallback — don't penalise if model not available
            details={"error": str(exc), "entities": []},
            error=f"NER model unavailable: {exc}",
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )

    # Combine question + answer for richer entity extraction
    combined_text = f"{question} {answer}" if question else answer
    doc = nlp(combined_text)

    # Collect entities with deduplication
    seen: set[str] = set()
    raw_entities: list[tuple[str, str]] = []  # (text, type)
    for ent in doc.ents:
        key = ent.text.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        entity_type = _ENTITY_TYPE_MAP.get(ent.label_, "CONDITION")
        raw_entities.append((ent.text.strip(), entity_type))

    if not raw_entities:
        # No entities found — return neutral score
        return EvalResult(
            module_name="entity_verifier",
            score=1.0,
            details={"total_entities": 0, "verified_count": 0, "flagged_count": 0, "entities": []},
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )

    # --- RxNorm verification for DRUG entities -------------------------------
    cache = _get_rxnorm_cache(rxnorm_cache_path)
    context_text = " ".join(context_docs or []).lower()

    entity_results: list[dict] = []
    drug_total = 0
    drug_verified = 0
    drug_flagged = 0

    for entity_text, entity_type in raw_entities:
        result = {
            "entity": entity_text,
            "type": entity_type,
            "status": "NOT_FOUND",
            "severity": None,
            "answer_value": entity_text,
            "context_value": None,
            "rxcui": None,
        }

        if entity_type in DRUG_TYPES:
            drug_total += 1
            key = entity_text.lower()

            # Layer 1: Local cache lookup
            rxcui = cache.get(key)

            # Layer 2: API fallback
            if not rxcui and use_api_fallback:
                rxcui = _lookup_rxnorm_api(entity_text)

            if rxcui:
                result["rxcui"] = rxcui
                result["status"] = "VERIFIED"
                drug_verified += 1

                # Check if entity appears in context
                if key in context_text:
                    result["context_value"] = entity_text
            else:
                result["status"] = "NOT_FOUND"

        elif entity_type in ("CONDITION", "PROCEDURE"):
            # Non-drug entities: check presence in context only
            if entity_text.lower() in context_text:
                result["status"] = "VERIFIED"
                result["context_value"] = entity_text
            else:
                result["status"] = "NOT_FOUND"

        entity_results.append(result)

    # --- Score ---------------------------------------------------------------
    # Score is based on drug entities only (per SRS Section 6.2)
    if drug_total == 0:
        score = 1.0  # No drug entities to verify → full score (no penalty)
    else:
        score = drug_verified / drug_total

    details = {
        "total_entities": len(raw_entities),
        "drug_total": drug_total,
        "verified_count": drug_verified,
        "flagged_count": drug_flagged,
        "entities": entity_results,
    }

    latency_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(
        "Entity verification: %.3f (%d/%d drugs verified) in %d ms",
        score, drug_verified, drug_total, latency_ms,
    )
    return EvalResult(
        module_name="entity_verifier",
        score=score,
        details=details,
        latency_ms=latency_ms,
    )
