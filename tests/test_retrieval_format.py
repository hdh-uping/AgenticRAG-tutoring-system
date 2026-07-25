import pytest
import torch

from app.tools import _binary_relevance_scores, _rrf_fusion, format_retrieval_results


def test_reranker_score_compares_yes_and_no_logits():
    logits = torch.zeros((2, 6))
    logits[0, 1], logits[0, 2] = 0.0, 2.0
    logits[1, 1], logits[1, 2] = 3.0, 1.0

    scores = _binary_relevance_scores(logits, no_id=1, yes_id=2)

    assert scores.tolist() == pytest.approx([0.880797, 0.119203], rel=1e-5)


def test_rrf_fusion_prioritizes_item_seen_by_both_retrievers():
    a = {"id": "a", "text": "A"}
    b = {"id": "b", "text": "B"}
    c = {"id": "c", "text": "C"}
    fused = _rrf_fusion([(a, 0.9), (b, 0.8)], [(b, 8.0), (c, 7.0)])
    assert fused[0]["id"] == "b"


def test_formatted_result_contains_stable_evidence_id_and_page():
    rendered = format_retrieval_results([{
        "id": "线性表_chunk_013",
        "text": "顺序表插入需要移动元素。",
        "page_num": 12,
        "header_path": "顺序表 > 插入",
        "rerank_score": 0.88,
    }])
    assert "id=线性表_chunk_013" in rendered
    assert "rerank_score=0.8800" in rendered
    assert "第12页" in rendered
