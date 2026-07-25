from kb.scripts.chunk_text import extract_inline_page_map, final_merge_small, find_page_num


def test_inline_page_markers_are_removed_and_mapped():
    text = "<!-- page:1 -->\n第一页有足够长的数组定义内容用于建立页码签名。" \
           "<!-- page:2 -->\n第二页有足够长的地址计算内容用于建立页码签名。"
    clean_text, page_map = extract_inline_page_map(text)
    assert "<!-- page:" not in clean_text
    assert find_page_num("第一页有足够长的数组定义内容用于建立页码签名。", page_map) == 1
    assert find_page_num("第二页有足够长的地址计算内容用于建立页码签名。", page_map) == 2


def test_tiny_document_title_is_merged_into_first_content_chunk():
    chunks = [
        {"text": "# 标题", "char_count": 4, "header_path": "## 正文"},
        {"text": "正文" * 120, "char_count": 240, "header_path": "## 第一节"},
    ]
    merged = final_merge_small(chunks)
    assert len(merged) == 1
    assert merged[0]["text"].startswith("# 标题\n\n正文")
