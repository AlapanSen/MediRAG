"""
FR-04: Vector Retrieval
=======================
FAISS IndexFlatIP with L2-normalised vectors (inner product = cosine similarity).
Returns top-k chunks as (chunk_text, metadata_dict, similarity_score) tuples.

Usage (as a module):
    from src.pipeline.retriever import Retriever
    r = Retriever(config)
    results = r.search("What is the treatment for Type 2 Diabetes?")
    for text, meta, score in results:
        print(score, meta["pub_type"], text[:80])

Usage (smoke test):
    python src/pipeline/retriever.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import logging
import pickle
from typing import Any

import faiss
import numpy as np
import yaml

logger = logging.getLogger(__name__)


class Retriever:
    """
    FAISS-based document retriever.

    Lazy-loads the embedding model and index on first search call.
    """

    def __init__(self, config: dict) -> None:
        self.config        = config
        self.top_k: int    = config["retrieval"]["top_k"]
        self.model_name: str = config["retrieval"]["embedding_model"]
        self.index_path: str = config["retrieval"]["index_path"]
        self.meta_path: str  = config["retrieval"]["metadata_path"]

        self._model    = None
        self._index    = None
        self._metadata: dict[int, dict] | None = None

    # ------------------------------------------------------------------
    # Private loaders (lazy)
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading BioBERT: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)

    def _load_index(self) -> None:
        if self._index is not None:
            return

        idx_path  = Path(self.index_path)
        meta_path = Path(self.meta_path)

        if not idx_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at '{idx_path}'. "
                "Run python src/pipeline/ingest.py && python src/pipeline/embedder.py first."
            )

        logger.info("Loading FAISS index from %s", idx_path)
        self._index = faiss.read_index(str(idx_path))

        logger.info("Loading metadata store from %s", meta_path)
        with open(meta_path, "rb") as f:
            self._metadata = pickle.load(f)

        logger.info(
            "Retriever ready: %d vectors, %d metadata entries",
            self._index.ntotal, len(self._metadata),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[tuple[str, dict[str, Any], float]]:
        """
        Search for the top-k most relevant chunks.

        Args:
            query : Natural language query
            top_k : Override config top_k if provided

        Returns:
            List of (chunk_text, metadata_dict, similarity_score),
            sorted by descending score.
        """
        if not query or not query.strip():
            logger.warning("Retriever.search called with empty query — returning []")
            return []

        k = top_k or self.top_k

        self._load_model()
        self._load_index()

        # Encode + normalise query
        q_vec: np.ndarray = self._model.encode(
            [query.strip()],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        # FAISS cosine search (inner product on normalised vecs)
        scores_arr, idx_arr = self._index.search(q_vec, k)
        scores = scores_arr[0]
        indices = idx_arr[0]

        results: list[tuple[str, dict, float]] = []
        for score, faiss_idx in zip(scores, indices):
            if faiss_idx == -1:          # FAISS padding for insufficient results
                continue
            meta = self._metadata.get(int(faiss_idx), {})
            text = meta.get("chunk_text", "")
            results.append((text, meta, float(score)))

        logger.debug(
            "Query '%s...' → %d results (top score=%.4f)",
            query[:40], len(results), results[0][2] if results else 0.0,
        )
        return results


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    import src  # noqa: F401 — logging
    config = _load_config()
    retriever = Retriever(config)

    test_queries = [
        "What is the recommended dosage of Metformin for Type 2 Diabetes in elderly patients?",
        "Contraindications of ibuprofen for patients with chronic kidney disease",
        "First-line treatment for hypertension according to clinical guidelines",
    ]

    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"QUERY: {query}")
        print("=" * 70)
        results = retriever.search(query, top_k=3)
        if not results:
            print("  No results — is the FAISS index built?")
            continue
        for rank, (text, meta, score) in enumerate(results, 1):
            print(f"\n  Rank {rank} | score={score:.4f} | source={meta.get('source')} | "
                  f"tier_type={meta.get('pub_type')}")
            print(f"  Title: {meta.get('title', '')[:80]}")
            print(f"  Text : {text[:200]}...")
