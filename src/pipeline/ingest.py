"""
FR-01: Document Ingestion
=========================
Loads documents from:
  - PubMedQA   (HuggingFace: pubmed_qa, pqa_labeled) — up to 500 samples
  - MedQA-USMLE (local JSONL from jind11/MedQA)       — up to 200 samples

Then calls chunker.py to split and saves chunks to data/processed/chunks.jsonl.

Usage:
    python src/pipeline/ingest.py
    python src/pipeline/ingest.py --pubmedqa 500 --medqa 200
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

# Make project root importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
import logging
import uuid
import yaml
from typing import Any

import src  # noqa: F401 — triggers logging setup

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# PubMedQA Ingestion (FR-01)
# ---------------------------------------------------------------------------

def ingest_pubmedqa(max_samples: int = 500) -> list[dict[str, Any]]:
    """
    Load PubMedQA from HuggingFace datasets.
    Each QA item contributes its context passages (abstracts) as documents,
    plus its long_answer if available.

    pub_type = "research_abstract" → Tier 3 (SRS FR-03b)
    """
    # Use 'pqa_artificial' (211k rows) if asking for more than 1000,
    # as 'pqa_labeled' only has 1000 rows.
    split_name = "pqa_artificial" if max_samples > 1000 else "pqa_labeled"
    logger.info("Loading PubMedQA split='%s' (max %d QA pairs)...", split_name, max_samples)
    try:
        from datasets import load_dataset
        dataset = load_dataset(
            "pubmed_qa", split_name, split="train", trust_remote_code=True
        )
    except Exception as exc:
        logger.error("Failed to load PubMedQA from HuggingFace: %s", exc)
        logger.error("Ensure you have an internet connection and datasets>=2.18.0")
        return []

    documents: list[dict] = []
    for i, item in enumerate(dataset):
        if i >= max_samples:
            break

        pub_id = str(item.get("pubid", uuid.uuid4().hex[:8]))
        question = item.get("question", "")[:200]

        # Index each context passage as a separate document
        contexts: list[str] = item.get("context", {}).get("contexts", [])
        for ctx in contexts:
            if ctx and ctx.strip():
                documents.append({
                    "text":     ctx.strip(),
                    "title":    question,
                    "doc_id":   f"pubmedqa_{pub_id}",
                    "source":   "pubmedqa",
                    "pub_type": "research_abstract",
                    "pub_year": 0,
                    "journal":  "",
                })

        # Also index the long_answer (gold-standard explanation)
        long_ans: str = item.get("long_answer", "").strip()
        if long_ans:
            documents.append({
                "text":     long_ans,
                "title":    question,
                "doc_id":   f"pubmedqa_{pub_id}_ans",
                "source":   "pubmedqa",
                "pub_type": "research_abstract",
                "pub_year": 0,
                "journal":  "",
            })

    logger.info(
        "PubMedQA: %d documents loaded from %d QA items",
        len(documents),
        min(max_samples, len(dataset)),
    )
    return documents


# ---------------------------------------------------------------------------
# MedQA-USMLE Ingestion (FR-01)
# ---------------------------------------------------------------------------

def ingest_medqa(
    data_dir: str = "data/raw/medqa",
    max_samples: int = 200,
) -> list[dict[str, Any]]:
    """
    Load MedQA-USMLE from local JSONL files.

    To obtain the data:
        git clone https://github.com/jind11/MedQA
        Copy the JSONL files from data_clean/questions/US/ to data/raw/medqa/

    pub_type = "exam_question" → Tier 5 (SRS FR-03b)
    """
    data_path = Path(data_dir)
    jsonl_files = sorted(list(data_path.glob("*.jsonl")) + list(data_path.glob("**/*.jsonl")))

    if not jsonl_files:
        logger.warning(
            "MedQA data not found at '%s'. "
            "To get it: git clone https://github.com/jind11/MedQA "
            "and copy JSONL files to %s/",
            data_dir, data_dir,
        )
        return []

    logger.info("Loading MedQA from '%s' (%d files)...", data_dir, len(jsonl_files))
    documents: list[dict] = []

    for jsonl_file in jsonl_files:
        if len(documents) >= max_samples:
            break
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for raw_line in f:
                if len(documents) >= max_samples:
                    break
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    item = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed JSON in %s: %s", jsonl_file.name, exc)
                    continue

                question:    str  = item.get("question", "")
                options:     dict = item.get("options", {})
                answer_key:  str  = item.get("answer", "")
                answer_text: str  = options.get(answer_key, "")

                # Combine question + all options + correct answer as document text
                opts_text = "  ".join(f"{k}: {v}" for k, v in options.items())
                text = f"Question: {question}\nOptions: {opts_text}"
                if answer_text:
                    text += f"\nAnswer ({answer_key}): {answer_text}"

                documents.append({
                    "text":     text,
                    "title":    question[:200],
                    "doc_id":   f"medqa_{uuid.uuid4().hex[:10]}",
                    "source":   "medqa",
                    "pub_type": "exam_question",
                    "pub_year": 0,
                    "journal":  "",
                })

    logger.info("MedQA: %d documents loaded", len(documents))
    return documents


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_raw_documents(documents: list[dict], output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    logger.info("Saved %d raw documents to %s", len(documents), output_path)


def _save_chunks(chunks: list[dict], output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    logger.info("Saved %d chunks to %s", len(chunks), output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="MediRAG-Eval Document Ingestion (FR-01)")
    parser.add_argument("--pubmedqa", type=int, default=500, help="Max PubMedQA samples")
    parser.add_argument("--medqa",    type=int, default=200, help="Max MedQA-USMLE samples")
    parser.add_argument(
        "--medqa-dir", default="data/raw/medqa",
        help="Directory containing MedQA JSONL files",
    )
    args = parser.parse_args()

    config = _load_config()

    # --- Ingest ---
    pubmedqa_docs = ingest_pubmedqa(max_samples=args.pubmedqa)
    medqa_docs    = ingest_medqa(data_dir=args.medqa_dir, max_samples=args.medqa)
    all_docs      = pubmedqa_docs + medqa_docs

    logger.info("Total documents ingested: %d", len(all_docs))

    if not all_docs:
        logger.error("No documents loaded. Check internet for PubMedQA and/or data/raw/medqa/ for MedQA.")
        sys.exit(1)

    # --- Save raw documents (for inspection) ---
    _save_raw_documents(all_docs, "data/raw/documents.jsonl")

    # --- Chunk ---
    from src.pipeline.chunker import chunk_documents
    chunks = chunk_documents(all_docs, config)
    logger.info("Total chunks produced: %d", len(chunks))

    # --- Save chunks for embedder ---
    _save_chunks(chunks, "data/processed/chunks.jsonl")

    logger.info("Ingestion complete. Now run: python src/pipeline/embedder.py")


if __name__ == "__main__":
    main()
