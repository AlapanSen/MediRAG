"""
FR-22: src/evaluate.py — MediRAG Evaluation Orchestrator
=========================================================
Top-level entry point for the evaluation pipeline.

Runs all 4 evaluation modules + RAGAS + aggregator for a given
(question, answer, context_docs) triple, returning a fully structured
composite EvalResult.

Usage as a module:
    from src.evaluate import run_evaluation
    result = run_evaluation(question, answer, context_docs)
    print(f"Score: {result.score:.3f} ({result.details['confidence_level']})")

Usage from CLI:
    python -m src.evaluate \\
        --question "What is the recommended dosage of Metformin for Type 2 Diabetes?" \\
        --answer "Metformin is typically started at 500mg twice daily..." \\
        --context-file data/processed/chunks.jsonl \\
        --top-k 5

SRS reference: FR-22, Section 7 (Evaluation Pipeline Overview)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

from src.modules.base import EvalResult
from src.modules.faithfulness import score_faithfulness
from src.modules.entity_verifier import verify_entities
from src.modules.source_credibility import score_source_credibility
from src.modules.contradiction import score_contradiction
from src.evaluation.ragas_eval import score_ragas
from src.evaluation.aggregator import aggregate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def run_evaluation(
    question: str,
    answer: str,
    context_chunks: list[dict],
    rxnorm_cache_path: str = "data/rxnorm_cache.csv",
    run_ragas: bool = True,
    weights: Optional[dict[str, float]] = None,
) -> EvalResult:
    """
    Run the full MediRAG evaluation pipeline for a single QA pair.

    Args:
        question          : Original user question.
        answer            : LLM-generated answer to evaluate.
        context_chunks    : List of retrieved chunk dicts (from retriever.retrieve()).
                           Each chunk must have at minimum {'text': str}.
        rxnorm_cache_path : Path to rxnorm_cache.csv for entity verification.
        run_ragas         : Whether to run the RAGAS module (requires LLM backend).
        weights           : Override default aggregation weights (optional).

    Returns:
        EvalResult for the "aggregator" module containing:
            .score          → composite score in [0, 1]
            .details        → full breakdown per module
            .latency_ms     → total wall-clock time
    """
    t_start = time.perf_counter()
    logger.info("=== MediRAG Evaluation START ===")
    logger.info("Question: %s", question[:120])
    logger.info("Answer  : %s", answer[:120])
    logger.info("Chunks  : %d context documents", len(context_chunks))

    # Extract text and metadata for modules that need it
    context_texts: list[str] = [c.get("text", "") for c in context_chunks]
    chunk_ids: list[str] = [
        c.get("chunk_id") or c.get("metadata", {}).get("chunk_id") or f"chunk_{i}"
        for i, c in enumerate(context_chunks)
    ]

    # -------------------------------------------------------------------------
    # Module 1: Faithfulness (DeBERTa NLI)
    # -------------------------------------------------------------------------
    logger.info("--- Module 1: Faithfulness ---")
    faith_result = score_faithfulness(
        answer=answer,
        context_docs=context_texts,
        chunk_ids=chunk_ids,
    )

    # -------------------------------------------------------------------------
    # Module 2: Entity Verification (SciSpaCy + RxNorm)
    # -------------------------------------------------------------------------
    logger.info("--- Module 2: Entity Verification ---")
    entity_result = verify_entities(
        answer=answer,
        question=question,
        context_docs=context_texts,
        rxnorm_cache_path=rxnorm_cache_path,
    )

    # -------------------------------------------------------------------------
    # Module 3: Source Credibility (Evidence Tier)
    # -------------------------------------------------------------------------
    logger.info("--- Module 3: Source Credibility ---")
    source_result = score_source_credibility(retrieved_chunks=context_chunks)

    # -------------------------------------------------------------------------
    # Module 4: Contradiction Detection (DeBERTa NLI cross-check)
    # -------------------------------------------------------------------------
    logger.info("--- Module 4: Contradiction Detection ---")
    contra_result = score_contradiction(
        answer=answer,
        context_docs=context_texts,
    )

    # -------------------------------------------------------------------------
    # RAGAS (optional — requires LLM backend)
    # -------------------------------------------------------------------------
    ragas_result: Optional[EvalResult] = None
    if run_ragas:
        logger.info("--- RAGAS Evaluation ---")
        ragas_result = score_ragas(
            question=question,
            answer=answer,
            context_docs=context_texts,
        )

    # -------------------------------------------------------------------------
    # Aggregator: weighted composite
    # -------------------------------------------------------------------------
    logger.info("--- Aggregator ---")
    agg_result = aggregate(
        faithfulness_result=faith_result,
        entity_result=entity_result,
        source_result=source_result,
        contradiction_result=contra_result,
        ragas_result=ragas_result,
        weights=weights,
    )

    total_ms = int((time.perf_counter() - t_start) * 1000)
    agg_result.details["total_pipeline_ms"] = total_ms

    # Attach per-module results for API/dashboard access
    agg_result.details["module_results"] = {
        "faithfulness":       {"score": faith_result.score,  "details": faith_result.details},
        "entity_verifier":    {"score": entity_result.score, "details": entity_result.details},
        "source_credibility": {"score": source_result.score, "details": source_result.details},
        "contradiction":      {"score": contra_result.score, "details": contra_result.details},
        "ragas":              {"score": ragas_result.score,  "details": ragas_result.details} if ragas_result else None,
    }

    logger.info(
        "=== MediRAG Evaluation DONE: score=%.3f (%s) in %d ms ===",
        agg_result.score,
        agg_result.details.get("confidence_level", "?"),
        total_ms,
    )
    return agg_result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MediRAG evaluation pipeline (FR-22)"
    )
    p.add_argument("--question",      required=True, help="User question")
    p.add_argument("--answer",        required=True, help="LLM answer to evaluate")
    p.add_argument("--context-file",  default="data/processed/chunks.jsonl",
                   help="JSONL file of chunks (output of ingest.py)")
    p.add_argument("--top-k",         type=int, default=5,
                   help="Number of context chunks to use")
    p.add_argument("--rxnorm-cache",  default="data/rxnorm_cache.csv",
                   help="Path to rxnorm_cache.csv")
    p.add_argument("--no-ragas",      action="store_true",
                   help="Skip RAGAS evaluation (no LLM backend needed)")
    p.add_argument("--json",          action="store_true",
                   help="Output result as JSON")
    return p


def _load_context_from_file(path: str, top_k: int) -> list[dict]:
    """Load top-k chunks from a JSONL file as simple dicts."""
    chunks = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
                if len(chunks) >= top_k:
                    break
    except FileNotFoundError:
        logger.error("Context file not found: %s", path)
        sys.exit(1)
    return chunks


if __name__ == "__main__":
    import yaml

    # Load config.yaml for logging setup
    try:
        cfg = yaml.safe_load(Path("config.yaml").read_text())
        log_level = cfg.get("logging", {}).get("level", "INFO")
    except Exception:
        log_level = "INFO"

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = _build_parser().parse_args()
    chunks = _load_context_from_file(args.context_file, args.top_k)

    result = run_evaluation(
        question=args.question,
        answer=args.answer,
        context_chunks=chunks,
        rxnorm_cache_path=args.rxnorm_cache,
        run_ragas=not args.no_ragas,
    )

    if args.json:
        import dataclasses
        print(json.dumps(dataclasses.asdict(result), indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"  MediRAG Evaluation Result")
        print(f"{'='*60}")
        print(f"  Score          : {result.score:.3f}")
        print(f"  Confidence     : {result.details.get('confidence_level', 'N/A')}")
        print(f"  Pipeline time  : {result.details.get('total_pipeline_ms', 0)} ms")
        print(f"\n  Module Breakdown:")
        for mod, res in (result.details.get("module_results") or {}).items():
            if res:
                print(f"    {mod:22s}: {res['score']:.3f}")
        print(f"{'='*60}\n")
