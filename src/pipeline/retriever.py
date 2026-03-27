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
    Hybrid FAISS + BM25 document retriever.

    On first search, lazily builds a BM25 index over all chunk texts.
    Each search runs both FAISS (semantic) and BM25 (keyword) then merges
    results using Reciprocal Rank Fusion (RRF) for best-of-both precision
    and recall.
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
        self._bm25     = None          # built lazily on first search
        self._bm25_ids: list[int] = [] # maps bm25 row → faiss_idx

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

    def _build_bm25(self) -> None:
        """Build BM25 index from the loaded metadata store (called once)."""
        if self._bm25 is not None:
            return
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank-bm25 not installed — falling back to FAISS-only. "
                           "Run: pip install rank-bm25")
            return

        logger.info("Building BM25 index over %d chunks…", len(self._metadata))
        corpus_ids: list[int] = []
        corpus_tokens: list[list[str]] = []
        for faiss_idx, meta in self._metadata.items():
            text = meta.get("chunk_text", "")
            if text:
                corpus_ids.append(faiss_idx)
                corpus_tokens.append(text.lower().split())

        self._bm25 = BM25Okapi(corpus_tokens)
        self._bm25_ids = corpus_ids
        logger.info("BM25 index ready (%d docs).", len(corpus_ids))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[tuple[str, dict[str, Any], float]]:
        """
        Hybrid semantic + keyword search using Reciprocal Rank Fusion.

        Args:
            query : Natural language query
            top_k : Override config top_k if provided

        Returns:
            List of (chunk_text, metadata_dict, rrf_score),
            sorted by descending combined score.
        """
        if not query or not query.strip():
            logger.warning("Retriever.search called with empty query — returning []")
            return []

        k = top_k or self.top_k
        # Fetch 3× more candidates from each retriever before fusion
        fetch_k = min(k * 3, 30)
        RRF_K = 60  # standard RRF constant (higher = smoother rank blending)

        self._load_model()
        self._load_index()
        self._build_bm25()

        # ── 1. FAISS semantic search ──────────────────────────────────
        q_vec: np.ndarray = self._model.encode(
            [query.strip()],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        scores_arr, idx_arr = self._index.search(q_vec, fetch_k)
        faiss_scores = scores_arr[0]
        faiss_indices = idx_arr[0]

        # Map faiss_idx → rank (1-indexed)
        faiss_ranks: dict[int, int] = {}
        for rank, (faiss_idx, score) in enumerate(zip(faiss_indices, faiss_scores), 1):
            if faiss_idx != -1:
                faiss_ranks[int(faiss_idx)] = rank

        # ── 2. BM25 keyword search ────────────────────────────────────
        bm25_ranks: dict[int, int] = {}
        if self._bm25 is not None:
            query_tokens = query.lower().split()
            bm25_scores_arr = self._bm25.get_scores(query_tokens)
            # Get top fetch_k indices by BM25 score
            top_bm25 = np.argsort(bm25_scores_arr)[::-1][:fetch_k]
            for rank, corpus_pos in enumerate(top_bm25, 1):
                if bm25_scores_arr[corpus_pos] > 0:
                    faiss_idx = self._bm25_ids[corpus_pos]
                    bm25_ranks[faiss_idx] = rank

        # ── 3. Reciprocal Rank Fusion ─────────────────────────────────
        # Score = 1/(k+rank_faiss) + 1/(k+rank_bm25)
        # A chunk only in FAISS gets 1/(60+rank); only in BM25 gets 1/(60+rank)
        # A chunk in BOTH gets the sum — it floats to the top
        all_ids = set(faiss_ranks.keys()) | set(bm25_ranks.keys())
        rrf_scores: dict[int, float] = {}
        for faiss_idx in all_ids:
            score = 0.0
            if faiss_idx in faiss_ranks:
                score += 1.0 / (RRF_K + faiss_ranks[faiss_idx])
            if faiss_idx in bm25_ranks:
                score += 1.0 / (RRF_K + bm25_ranks[faiss_idx])
            rrf_scores[faiss_idx] = score

        # Sort by RRF score descending, take top-k
        top_ids = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[:k]

        results: list[tuple[str, dict, float]] = []
        for faiss_idx in top_ids:
            meta = self._metadata.get(faiss_idx, {})
            text = meta.get("chunk_text", "")
            results.append((text, meta, rrf_scores[faiss_idx]))

        logger.debug(
            "Hybrid query '%s...' → %d results (top RRF=%.4f) "
            "[FAISS candidates: %d, BM25 candidates: %d]",
            query[:40], len(results),
            results[0][2] if results else 0.0,
            len(faiss_ranks), len(bm25_ranks),
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
