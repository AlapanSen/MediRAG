import pytest
from httpx import Client, ASGITransport
from src.api.main import app

# Use modern httpx transport to avoid Starlette/FastAPI TestClient version conflicts
client = Client(transport=ASGITransport(app=app), base_url="http://test")

def test_health_endpoint():
    """Test that the /health endpoint correctly reports system status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "retriever_ready" in data

def test_evaluate_endpoint():
    """Test the /evaluate endpoint with mock claims."""
    payload = {
        "question": "Is Metformin safe?",
        "answer": "Metformin is a safe and effective drug. It is recommended.",
        "context_chunks": [
            {
                "chunk_id": "mock-1",
                "text": "Metformin is a first-line medication for the treatment of type 2 diabetes. It is safe.",
                "source": "mock_db",
                "pub_type": "research_abstract",
                "pub_year": 2024,
                "title": "Study on Metformin safety"
            }
        ],
        "run_ragas": False
    }

    # Since the evaluation modules load heavy ML models, 
    # the first test call might take 10-15s to run.
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "composite_score" in data
    assert "hrs" in data
    assert data["risk_band"] in ["LOW", "MODERATE", "HIGH", "CRITICAL"]
    assert "faithfulness" in data["module_results"]

def test_query_invalid_params():
    """Test the /query validation rules."""
    payload = {
        "question": "Short",  # Too short, should fail validation (min 5 chars)
        "top_k": 5
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 422  # Unprocessable Entity (Pydantic validation error)
