"""
src/scripts/warmup.py
=====================
Pre-loads heavy ML models (FAISS, DeBERTa, SciSpaCy) into memory 
and guarantees instantaneous responses for the first API request during the live demo.

Usage:
    python scripts/warmup.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import time
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("warmup")

def main():
    api_url = "http://localhost:8000"
    
    logger.info("Verifying API is running...")
    try:
        health = requests.get(f"{api_url}/health", timeout=5)
        health.raise_for_status()
        logger.info(f"API Health: {health.json()}")
    except requests.exceptions.RequestException as e:
        logger.error(f"API is not running at {api_url}. Please start it with 'uvicorn src.api.main:app' first.")
        sys.exit(1)

    logger.info("Sending WARMUP query to load DeBERTa, SciSpaCy, and FAISS into RAM... (This may take 15-25s)")
    t0 = time.time()
    
    # We send a basic query to force all models to initialize
    payload = {
        "question": "What is the recommended dosage of Metformin for elderly Type 2 Diabetes patients?",
        "top_k": 1,
        "run_ragas": False
    }
    
    try:
        resp = requests.post(f"{api_url}/query", json=payload, timeout=60)
        resp.raise_for_status()
        elapsed = time.time() - t0
        logger.info(f"Warmup successful in {elapsed:.1f}s!")
        logger.info("All machine learning models are now cached in RAM.")
        logger.info("The next API requests will be completely instantaneous.")
    except Exception as e:
        logger.error(f"Warmup failed: {e}")
        if hasattr(e, "response") and e.response is not None:
            logger.error(f"Response: {e.response.text}")

if __name__ == "__main__":
    main()
