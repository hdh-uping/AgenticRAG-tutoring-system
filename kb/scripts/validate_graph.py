"""验证当前三类实体和三类关系的 JSON 图谱一致性。"""
import json
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent.parent
GRAPH_DIR = BASE_DIR / "kb" / "graph"
ENTITY_TYPES = {"数据结构", "操作", "复杂度"}
RELATION_TYPES = {"IS_A", "HAS_OPERATION", "HAS_COMPLEXITY"}
RELATION_ENDPOINT_TYPES = {
    "IS_A": ("数据结构", "数据结构"),
    "HAS_OPERATION": ("数据结构", "操作"),
    "HAS_COMPLEXITY": ("操作", "复杂度"),
}


def validate_graph(
    entities_path: Path = GRAPH_DIR / "entities.json",
    relations_path: Path = GRAPH_DIR / "relations.json",
    content_files: list[Path] | None = None,
) -> dict:
    entities = json.loads(entities_path.read_text(encoding="utf-8"))
    relations = json.loads(relations_path.read_text(encoding="utf-8"))
    errors = []
    warnings = []

    names = [entity.get("name", "") for entity in entities]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        errors.append(f"实体名称重复: {', '.join(duplicate_names)}")

    entity_by_name = {entity.get("name"): entity for entity in entities if entity.get("name")}
    for index, entity in enumerate(entities):
        name = entity.get("name")
        entity_type = entity.get("type")
        if not name:
            errors.append(f"实体 #{index} 缺少 name")
        if entity_type not in ENTITY_TYPES:
            errors.append(f"实体 {name or index} 类型非法: {entity_type}")
        if entity_type in {"数据结构", "操作"} and not entity.get("description"):
            warnings.append(f"实体 {name} 缺少 description")

    relation_keys = []
    connected = set()
    for index, relation in enumerate(relations):
        head = relation.get("head")
        tail = relation.get("tail")
        relation_type = relation.get("relation")
        key = (head, relation_type, tail)
        relation_keys.append(key)

        if relation_type not in RELATION_TYPES:
            errors.append(f"关系 #{index} 类型非法: {relation_type}")
            continue
        if head not in entity_by_name:
            errors.append(f"关系 #{index} 的起点不存在: {head}")
        if tail not in entity_by_name:
            errors.append(f"关系 #{index} 的终点不存在: {tail}")
        if head in entity_by_name and tail in entity_by_name:
            expected_head, expected_tail = RELATION_ENDPOINT_TYPES[relation_type]
            actual_head = entity_by_name[head].get("type")
            actual_tail = entity_by_name[tail].get("type")
            if (actual_head, actual_tail) != (expected_head, expected_tail):
                errors.append(
                    f"关系 {head}-[{relation_type}]->{tail} 端点类型错误: "
                    f"{actual_head}->{actual_tail}"
                )
            connected.update((head, tail))

    duplicate_relations = sorted({key for key in relation_keys if relation_keys.count(key) > 1})
    if duplicate_relations:
        errors.append(f"存在 {len(duplicate_relations)} 条重复关系")

    isolated = sorted(set(entity_by_name) - connected)
    if isolated:
        warnings.append(f"孤立实体: {', '.join(isolated)}")

    nonstandard_provenance = sorted(
        entity["name"]
        for entity in entities
        if entity.get("source_chunks") and not entity.get("content_ids")
    )
    if nonstandard_provenance:
        warnings.append(
            f"{len(nonstandard_provenance)} 个实体使用 source_chunks 而非标准 content_ids，"
            f"来源链不一致: {', '.join(nonstandard_provenance)}"
        )

    missing_content_ids = sorted(
        entity["name"] for entity in entities if not entity.get("content_ids")
    )
    malformed_content_ids = sorted({
        str(content_id)
        for entity in entities
        for content_id in entity.get("content_ids", [])
        if not re.fullmatch(r".+:content:\d{4,}", str(content_id))
    })
    if missing_content_ids:
        warnings.append(
            f"{len(missing_content_ids)} 个实体缺少标准 content_ids: "
            f"{', '.join(missing_content_ids)}"
        )
    if malformed_content_ids:
        warnings.append(
            f"存在 {len(malformed_content_ids)} 个非标准 content_id: "
            f"{', '.join(malformed_content_ids[:10])}"
        )

    unresolved_content_ids = []
    if content_files:
        valid_content_ids = set()
        for content_file in content_files:
            content = json.loads(content_file.read_text(encoding="utf-8"))
            doc_name = content_file.stem.removesuffix("_content")
            valid_content_ids.update(
                f"{doc_name}:content:{index:04d}" for index in range(len(content))
            )
        unresolved_content_ids = sorted({
            str(content_id)
            for entity in entities
            for content_id in entity.get("content_ids", [])
            if str(content_id) not in valid_content_ids
        })
        if unresolved_content_ids:
            errors.append(
                f"存在 {len(unresolved_content_ids)} 个无法解析到 MinerU content.json 的来源 ID: "
                f"{', '.join(unresolved_content_ids[:10])}"
            )

    counts = {
        "entities": len(entities),
        "relations": len(relations),
        "entity_types": {
            entity_type: sum(entity.get("type") == entity_type for entity in entities)
            for entity_type in sorted(ENTITY_TYPES)
        },
        "relation_types": {
            relation_type: sum(item.get("relation") == relation_type for item in relations)
            for relation_type in sorted(RELATION_TYPES)
        },
    }
    return {
        "valid": not errors,
        "provenance_consistent": not (
            nonstandard_provenance or missing_content_ids or malformed_content_ids
            or unresolved_content_ids
        ),
        "errors": errors,
        "warnings": warnings,
        "counts": counts,
    }


def main() -> int:
    report = validate_graph()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
