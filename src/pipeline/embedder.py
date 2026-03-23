"""
FR-03 + FR-03b: Embedding Generation & FAISS Index Construction
===============================================================
Model  : dmis-lab/biobert-v1.1  (768-dim dense vectors, SentenceTransformer)
Index  : FAISS IndexFlatIP with L2-normalized vectors  (= cosine similarity)
Metadata: Parallel dict[int → dict] saved as pickle alongside index

Usage:
    python src/pipeline/embedder.py
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import logging
import pickle

import faiss
import numpy as np
import yaml

import src  # noqa: F401 — logging setup

logger = logging.getLogger(__name__)


def _load_config() -> dict:
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_chunks(chunks_path: str = "data/processed/chunks.jsonl") -> list[dict]:
    """Load chunks from JSONL produced by ingest.py."""
    path = Path(chunks_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: '{chunks_path}'. "
            "Run python src/pipeline/ingest.py first."
        )
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    logger.info("Loaded %d chunks from %s", len(chunks), chunks_path)
    return chunks


def encode_texts(
    texts: list[str],
    model_name: str,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Encode texts using BioBERT via SentenceTransformer.
    Returns L2-normalized float32 array of shape (N, 768).
    """
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)

    logger.info("Encoding %d texts (batch_size=%d)...", len(texts), batch_size)
    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2-normalise → cosine via IndexFlatIP
        convert_to_numpy=True,
    )
    logger.info("Encoded shape: %s", embeddings.shape)
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build FAISS IndexFlatIP.
    Because vectors are L2-normalised, inner product == cosine similarity.
    """
    dim = embeddings.shape[1]  # 768 for BioBERT
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info(
        "FAISS IndexFlatIP built: %d vectors, dim=%d", index.ntotal, dim
    )
    return index


def build_metadata_store(chunks: list[dict]) -> dict[int, dict]:
    """
    Build parallel metadata dict  →  key = FAISS integer index (0-based).
    Stores the full FR-03b schema plus chunk_text for retrieval.
    """
    store: dict[int, dict] = {}
    for i, chunk in enumerate(chunks):
        store[i] = {
            "chunk_id":     chunk["chunk_id"],
            "doc_id":       chunk["doc_id"],
            "source":       chunk["source"],
            "title":        chunk["title"],
            "pub_type":     chunk["pub_type"],
            "pub_year":     chunk["pub_year"],
            "journal":      chunk["journal"],
            "chunk_index":  chunk["chunk_index"],
            "total_chunks": chunk["total_chunks"],
            "chunk_text":   chunk["chunk_text"],   # kept for retrieval
        }
    return store


def save_artifacts(
    index: faiss.IndexFlatIP,
    metadata_store: dict,
    config: dict,
) -> None:
    """Persist FAISS index and metadata pickle to disk."""
    index_path = Path(config["retrieval"]["index_path"])
    meta_path  = Path(config["retrieval"]["metadata_path"])

    index_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_path))
    logger.info("FAISS index written to %s", index_path)

    with open(meta_path, "wb") as f:
        pickle.dump(metadata_store, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(
        "Metadata store written to %s (%d entries)", meta_path, len(metadata_store)
    )


def main() -> None:
    config = _load_config()
    chunks = load_chunks("data/processed/chunks.jsonl")

    if not chunks:
        logger.error("No chunks to embed. Run python src/pipeline/ingest.py first.")
        sys.exit(1)

    texts          = [c["chunk_text"] for c in chunks]
    model_name     = config["retrieval"]["embedding_model"]
    embeddings     = encode_texts(texts, model_name, batch_size=32)
    index          = build_faiss_index(embeddings)
    metadata_store = build_metadata_store(chunks)

    save_artifacts(index, metadata_store, config)

    logger.info(
        "Embedding complete. Index has %d vectors. "
        "Next: python scripts/warmup.py && streamlit run src/dashboard/app.py",
        index.ntotal,
    )


if __name__ == "__main__":
    main()
