"""
FR-02: Document Chunking
========================
LangChain RecursiveCharacterTextSplitter
  chunk_size = 512 chars  (config: retrieval.chunk_size)
  overlap    = 50  chars  (config: retrieval.chunk_overlap)

Each chunk carries the full FR-03b metadata schema required by Module 3
(source credibility) and the FAISS metadata store.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)


def chunk_documents(
    documents: list[dict[str, Any]],
    config: dict,
) -> list[dict[str, Any]]:
    """
    Split a list of raw documents into overlapping text chunks.

    Args:
        documents : List of dicts with keys:
                    text, doc_id, source, title, pub_type, pub_year, journal
        config    : Loaded config.yaml dict

    Returns:
        List of chunk dicts (FR-03b metadata schema):
            chunk_id, chunk_text, doc_id, source, title,
            pub_type, pub_year, journal, chunk_index, total_chunks
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    chunk_size    = config["retrieval"]["chunk_size"]      # 512
    chunk_overlap = config["retrieval"]["chunk_overlap"]   # 50

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: list[dict] = []

    for doc in documents:
        text = doc.get("text", "").strip()
        if not text:
            logger.debug("Skipping empty document: doc_id=%s", doc.get("doc_id"))
            continue

        raw_chunks = splitter.split_text(text)
        total = len(raw_chunks)

        for idx, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            all_chunks.append({
                # FR-03b schema
                "chunk_id":     str(uuid.uuid4()),
                "chunk_text":   chunk_text,
                "doc_id":       doc["doc_id"],
                "source":       doc["source"],
                "title":        doc["title"],
                "pub_type":     doc["pub_type"],
                "pub_year":     doc.get("pub_year", 0),
                "journal":      doc.get("journal", ""),
                "chunk_index":  idx,
                "total_chunks": total,
            })

    logger.info(
        "Chunked %d documents → %d chunks (size=%d, overlap=%d)",
        len(documents), len(all_chunks), chunk_size, chunk_overlap,
    )
    return all_chunks
