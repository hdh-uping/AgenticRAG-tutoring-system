from app.evidence import EvidencePool, append_references


def test_collects_and_deduplicates_chunk_evidence():
    pool = EvidencePool()
    observation = (
        "[Chunk1 | id=线性表_chunk_003 | rerank_score=0.9000 | 第1页 | 定义]\n内容"
    )
    assert len(pool.add_observation("hybrid_retrieval", observation)) == 1
    assert pool.add_observation("hybrid_retrieval", observation) == []
    assert pool.has_evidence


def test_collects_graph_node_and_appends_references():
    pool = EvidencePool()
    pool.add_observation("graph_lookup", "[操作] 顺序表·插入\n  时间复杂度: O(n)")
    answer = append_references("插入通常是 O(n)。", pool)
    assert "图谱节点：操作「顺序表·插入」" in answer


def test_model_generated_source_section_is_replaced():
    pool = EvidencePool()
    pool.add_observation(
        "hybrid_retrieval",
        "[Chunk1 | id=线性表_chunk_003 | rerank_score=0.9000 | 第1页]\n内容",
    )
    answer = append_references("回答。\n\n## 参考来源\n\n- `伪造_chunk`", pool)
    assert "伪造_chunk" not in answer
    assert "线性表_chunk_003" in answer


def test_only_explicitly_used_sources_are_selected():
    pool = EvidencePool()
    pool.add_observation(
        "hybrid_retrieval",
        "[Chunk1 | id=线性表_chunk_001 | rerank_score=0.9900 | 第1页]\n甲\n"
        "[Chunk2 | id=线性表_chunk_002 | rerank_score=0.9800 | 第2页]\n乙",
    )
    selected = pool.select_for_answer("结论来自线性表_chunk_002。")
    assert [source["id"] for source in selected] == ["线性表_chunk_002"]


def test_invalid_zero_page_is_not_presented_as_page_zero():
    pool = EvidencePool()
    pool.add_observation(
        "hybrid_retrieval",
        "[Chunk1 | id=线性表_chunk_017 | rerank_score=0.9000 | 第0页]\n内容",
    )
    source = pool.to_sources()[0]
    assert source["page_num"] is None
    assert "页码未标注" in append_references("回答。", pool)


def test_graph_source_is_kept_when_answer_explicitly_cites_a_chunk():
    pool = EvidencePool()
    pool.add_observation("graph_lookup", "[操作] 顺序表·插入\n  代码: ...")
    pool.add_observation(
        "hybrid_retrieval",
        "[Chunk1 | id=线性表_chunk_019 | rerank_score=0.9000 | 第4页]\n内容",
    )
    selected = pool.select_for_answer("参考线性表_chunk_019。")
    assert [source["id"] for source in selected] == ["顺序表·插入", "线性表_chunk_019"]


def test_top_chunk_is_kept_when_answer_only_mentions_graph_node():
    pool = EvidencePool()
    pool.add_observation("graph_lookup", "[数据结构] 稀疏矩阵\n  描述: 非零元素很少")
    pool.add_observation(
        "hybrid_retrieval",
        "[Chunk1 | id=数组_chunk_022 | rerank_score=0.9900 | 第4页]\n结构定义",
    )
    selected = pool.select_for_answer("稀疏矩阵可以使用三元组表示。")
    assert [source["id"] for source in selected] == ["稀疏矩阵", "数组_chunk_022"]
