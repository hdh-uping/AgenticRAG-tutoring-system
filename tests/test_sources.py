from app.sources import get_source, load_sources


def test_loads_all_current_chunks():
    assert len(load_sources()) == 266


def test_get_source_returns_full_metadata():
    source = get_source("线性表_chunk_003")
    assert source is not None
    assert source["document"] == "线性表"
    assert source["page_num"] == 1
    assert "线性表" in source["text"]


def test_unknown_source_returns_none():
    assert get_source("不存在的_chunk") is None


def test_array_source_is_available():
    source = get_source("数组_chunk_022")
    assert source is not None
    assert source["document"] == "数组"
    assert source["page_num"] == 4
