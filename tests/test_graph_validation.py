from kb.scripts.validate_graph import validate_graph


def test_current_graph_is_structurally_valid():
    report = validate_graph()
    assert report["valid"], report["errors"]
    assert report["provenance_consistent"] is True
    assert not any("source_chunks" in warning for warning in report["warnings"])
    assert report["counts"]["entities"] == 96
    assert report["counts"]["relations"] == 116
    assert report["counts"]["entity_types"] == {
        "复杂度": 3,
        "操作": 69,
        "数据结构": 24,
    }
