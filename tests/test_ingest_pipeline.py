import json

import pytest

import kb.scripts.build_graph as build_graph
from kb.scripts.build_graph import load_global_contents
from kb.scripts.ingest_pipeline import PipelineError, validate_doc_name
from kb.scripts.validate_graph import validate_graph


def test_document_content_ids_are_stable_and_document_scoped(tmp_path):
    first = tmp_path / "文档A_content.json"
    second = tmp_path / "文档B_content.json"
    first.write_text(json.dumps([{"text": "a"}, {"text": "b"}]), encoding="utf-8")
    second.write_text(json.dumps([{"text": "c"}]), encoding="utf-8")

    contents, all_content = load_global_contents([first, second])

    assert [item["idx"] for item in contents[first]] == [
        "文档A:content:0000",
        "文档A:content:0001",
    ]
    assert all_content[-1]["idx"] == "文档B:content:0000"


def test_graph_validator_resolves_content_ids(tmp_path):
    content = tmp_path / "数组_content.json"
    entities = tmp_path / "entities.json"
    relations = tmp_path / "relations.json"
    content.write_text(json.dumps([{"text": "数组"}]), encoding="utf-8")
    entities.write_text(
        json.dumps([
            {
                "name": "数组",
                "type": "数据结构",
                "description": "按下标访问的数据结构",
                "content_ids": ["数组:content:0000"],
            }
        ]),
        encoding="utf-8",
    )
    relations.write_text("[]", encoding="utf-8")

    report = validate_graph(entities, relations, [content])

    assert report["valid"] is True
    assert report["provenance_consistent"] is True


def test_document_name_rejects_path_traversal():
    with pytest.raises(PipelineError):
        validate_doc_name("../数组")


def test_synonym_merge_cannot_collapse_structures_or_cross_operation_prefixes(monkeypatch):
    class FakeCompletions:
        @staticmethod
        def create(**_kwargs):
            content = json.dumps({
                "merges": [
                    {"primary": "线性表", "aliases": ["顺序表"]},
                    {"primary": "单链表·删除", "aliases": ["单链表·删除节点", "顺序表·删除"]},
                ]
            })
            message = type("Message", (), {"content": content})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice]})()

    fake_client = type(
        "Client", (),
        {"chat": type("Chat", (), {"completions": FakeCompletions()})()},
    )()
    monkeypatch.setattr(build_graph, "get_llm_client", lambda: fake_client)
    entities = [
        {"name": "线性表", "type": "数据结构", "content_ids": ["a:content:0000"]},
        {"name": "顺序表", "type": "数据结构", "content_ids": ["a:content:0001"]},
        {"name": "单链表·删除", "type": "操作", "content_ids": ["a:content:0002"]},
        {"name": "单链表·删除节点", "type": "操作", "content_ids": ["a:content:0003"]},
        {"name": "顺序表·删除", "type": "操作", "content_ids": ["a:content:0004"]},
    ]

    merged, _ = build_graph.merge_across_batches(entities, [])
    names = {entity["name"] for entity in merged}

    assert {"线性表", "顺序表", "顺序表·删除"} <= names
    assert "单链表·删除节点" not in names
