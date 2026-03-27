import pytest
from src.modules.faithfulness import score_faithfulness
from src.modules.source_credibility import score_source_credibility
from src.modules.contradiction import score_contradiction
from src.evaluation.aggregator import aggregate

def test_source_credibility():
    chunks = [
        {"chunk_id": "c1", "pub_type": "research_abstract", "title": "Mock Paper"},
        {"chunk_id": "c2", "pub_type": "exam_question", "title": "Mock Exam Q"}
    ]
    results = score_source_credibility(chunks)
    assert results.score > 0.0
    assert 0.3 <= results.score <= 0.5
    assert results.details["chunk_count"] == 2

def test_faithfulness_nli():
    res_entail = score_faithfulness(
        answer="The sky is blue.",
        context_docs=["The sky is colored blue today."]
    )
    assert res_entail.score >= 0.8
    
    res_contra = score_faithfulness(
        answer="The sky is red.",
        context_docs=["The sky is completely blue and not red."]
    )
    assert res_contra.score <= 0.2

def test_aggregator_logic():
    # Mock config
    test_cfg = {
        "evaluation": {
            "weights": {
                "faithfulness": 0.4,
                "entity_accuracy": 0.2,
                "source_credibility": 0.2,
                "contradiction_risk": 0.2,
                "ragas_composite": 0.0
            }
        }
    }
    
    module_results = {
        "faithfulness": {"score": 1.0},
        "entity_verifier": {"score": 1.0},
        "source_credibility": {"score": 0.5},
        "contradiction": {"score": 1.0},
    }
    
    class MockResult:
        def __init__(self, score, error=None):
            self.score = score
            self.error = error
            self.latency_ms = 10
            
    res = aggregate(
        faithfulness_result=MockResult(1.0),
        entity_result=MockResult(1.0),
        source_result=MockResult(0.5),
        contradiction_result=MockResult(1.0),
        weights=test_cfg["evaluation"]["weights"]
    )
    assert abs(res.score - 0.9) < 0.01
    assert res.details["hrs"] == 10
    assert res.details["risk_band"] == "LOW"
