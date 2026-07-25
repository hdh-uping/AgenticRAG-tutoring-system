from app.local_graph import lookup, match_entity, related_concepts


def test_matches_operation_without_middle_dot():
    entity = match_entity("顺序表插入")
    assert entity is not None
    assert entity["name"] == "顺序表·插入"


def test_matches_specific_entity_when_query_contains_extra_words():
    entity = match_entity("稀疏矩阵压缩")
    assert entity is not None
    assert entity["name"] == "稀疏矩阵"


def test_lookup_returns_operation_complexity():
    result = lookup("顺序表插入")
    assert "[操作] 顺序表·插入" in result
    assert "时间复杂度" in result


def test_related_concepts_excludes_current_concept():
    result = related_concepts(["顺序表·插入"])
    assert "顺序表·插入" not in result
    assert result


def test_related_concepts_balances_multiple_input_concepts():
    result = related_concepts(["队列", "栈"])
    assert any(name.startswith("队列·") for name in result)
    assert any(name.startswith("栈·") for name in result)
