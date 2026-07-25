"""仓库内 JSON 图谱的只读查询降级实现。"""
import json
import re
from functools import lru_cache
from pathlib import Path


GRAPH_DIR = Path(__file__).resolve().parent.parent / "kb" / "graph"


@lru_cache(maxsize=1)
def _load_graph_data() -> tuple[list[dict], list[dict]]:
    entities = json.loads((GRAPH_DIR / "entities.json").read_text(encoding="utf-8"))
    relations = json.loads((GRAPH_DIR / "relations.json").read_text(encoding="utf-8"))
    return entities, relations


def is_available() -> bool:
    return (GRAPH_DIR / "entities.json").exists() and (GRAPH_DIR / "relations.json").exists()


def _normalize(value: str) -> str:
    return re.sub(r"[·\s]", "", value).lower()


def match_entity(concept: str) -> dict | None:
    entities, _ = _load_graph_data()
    raw = concept.strip()
    normalized = _normalize(raw)

    for entity in entities:
        names = [entity.get("name", ""), *entity.get("aliases", [])]
        if raw in names or normalized in {_normalize(name) for name in names if name}:
            return entity

    candidates = []
    for entity in entities:
        entity_name = _normalize(entity.get("name", ""))
        if normalized and entity_name and (
            normalized in entity_name or entity_name in normalized
        ):
            candidates.append(entity)
    # 查询包含节点名时优先最长实体，如“稀疏矩阵压缩”应命中“稀疏矩阵”而非“矩阵”。
    return max(candidates, key=lambda item: (len(item.get("name", "")), item.get("name", ""))) \
        if candidates else None


def _targets(relations: list[dict], head: str, relation: str) -> list[str]:
    return sorted({
        item["tail"] for item in relations
        if item.get("head") == head and item.get("relation") == relation
    })


def _heads(relations: list[dict], tail: str, relation: str) -> list[str]:
    return sorted({
        item["head"] for item in relations
        if item.get("tail") == tail and item.get("relation") == relation
    })


def lookup(concept: str) -> str:
    entity = match_entity(concept)
    if not entity:
        return f"(本地图谱中未找到与「{concept}」相关的节点。)"

    _, relations = _load_graph_data()
    name = entity["name"]
    label = entity.get("type", "未知")
    parts = [f"[{label}] {name}", "  数据源: 本地 JSON 图谱（Neo4j 降级）"]

    if entity.get("code"):
        parts.append("  ℹ️  以上已包含完整代码和步骤，可直接用于回答。")
    for field, field_cn in (("description", "描述"), ("pseudocode", "步骤"), ("code", "代码")):
        value = str(entity.get(field) or "")
        if value:
            parts.append(f"  {field_cn}: {value[:600]}{'...' if len(value) > 600 else ''}")

    if label == "数据结构":
        operations = _targets(relations, name, "HAS_OPERATION")
        children = _heads(relations, name, "IS_A")
        parents = _targets(relations, name, "IS_A")
        if operations:
            parts.append(f"  操作列表: {', '.join(operations)}")
        if children:
            parts.append(f"  子类型: {', '.join(children)}")
        if parents:
            parts.append(f"  属于: {', '.join(parents)}")
    elif label == "操作":
        complexities = _targets(relations, name, "HAS_COMPLEXITY")
        if complexities:
            parts.append(f"  时间复杂度: {', '.join(complexities)}")

    return "\n".join(parts)


def related_concepts(concepts: list[str], limit: int = 6) -> list[str]:
    _, relations = _load_graph_data()
    matched = []
    for concept in concepts:
        entity = match_entity(concept)
        if entity and entity["name"] not in matched:
            matched.append(entity["name"])

    candidate_groups = []
    for name in matched:
        # 同级操作。
        owners = _heads(relations, name, "HAS_OPERATION")
        groups = []
        groups.append([
            candidate
            for owner in owners
            for candidate in _targets(relations, owner, "HAS_OPERATION")
        ])
        # 数据结构自身的操作。
        groups.append(_targets(relations, name, "HAS_OPERATION"))
        # 兄弟结构。
        parents = _targets(relations, name, "IS_A")
        groups.append([
            candidate
            for parent in parents
            for candidate in _heads(relations, parent, "IS_A")
        ])
        # 父类型。
        groups.append(parents)

        candidates = []
        for group in groups:
            for candidate in group:
                if candidate not in matched and candidate not in candidates:
                    candidates.append(candidate)
        candidate_groups.append(candidates)

    # 多概念问题采用轮询，避免第一个概念独占全部候选名额。
    related = []
    max_group_size = max((len(group) for group in candidate_groups), default=0)
    for index in range(max_group_size):
        for group in candidate_groups:
            if index < len(group) and group[index] not in related:
                related.append(group[index])
                if len(related) >= limit:
                    return related
    return related
