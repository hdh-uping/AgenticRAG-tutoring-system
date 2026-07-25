"""related_concepts Skill。

本 Skill 只负责确定性地产生图谱候选；独立推荐 Agent 负责结合语境筛选和解释。
"""
import logging

from app.config import get_settings
from app.local_graph import related_concepts as local_related_concepts
from app.tools import _load_graph


logger = logging.getLogger(__name__)


def _query_names(session, query: str, *, name: str) -> list[str]:
    return [row[0] for row in session.run(query, name=name).values()]


def _find_related(driver, node_name: str) -> list[str]:
    """按“同级操作 → 自身操作 → 兄弟结构 → 父类型”的优先级查询。"""
    with driver.session() as session:
        groups = [
            _query_names(
                session,
                "MATCH (ds)-[:HAS_OPERATION]->(op {name: $name}) "
                "MATCH (ds)-[:HAS_OPERATION]->(sibling) "
                "RETURN sibling.name ORDER BY sibling.name",
                name=node_name,
            ),
            _query_names(
                session,
                "MATCH (ds {name: $name})-[:HAS_OPERATION]->(op) "
                "RETURN op.name ORDER BY op.name",
                name=node_name,
            ),
            _query_names(
                session,
                "MATCH (parent)<-[:IS_A]-(child {name: $name}) "
                "MATCH (parent)<-[:IS_A]-(sibling) "
                "RETURN sibling.name ORDER BY sibling.name",
                name=node_name,
            ),
            _query_names(
                session,
                "MATCH (child {name: $name})-[:IS_A]->(parent) "
                "RETURN parent.name ORDER BY parent.name",
                name=node_name,
            ),
        ]

    ordered = []
    for group in groups:
        for candidate in group:
            if candidate != node_name and candidate not in ordered:
                ordered.append(candidate)
    return ordered


def _match_name(driver, concept: str) -> str:
    with driver.session() as session:
        exact = list(session.run(
            "MATCH (n {name: $name}) RETURN n.name LIMIT 1", name=concept
        ))
        if exact:
            return exact[0]["n.name"]

        fuzzy = list(session.run(
            "MATCH (n) WHERE n.name CONTAINS $name "
            "RETURN n.name ORDER BY size(n.name), n.name LIMIT 1",
            name=concept,
        ))
        return fuzzy[0]["n.name"] if fuzzy else ""


def run(concepts: list[str], student_id: str = "", rating: int = 0) -> str:
    del student_id, rating  # 保留参数以兼容旧调用方。
    concepts = [c.strip() for c in concepts if c.strip()]
    if not concepts:
        return ""

    settings = get_settings()
    if not settings.neo4j_password:
        related = local_related_concepts(concepts)
        return f"你可能还想了解：{'、'.join(related)}。" if related else ""

    try:
        return _run_neo4j(concepts)
    except Exception:
        logger.exception("Neo4j 推荐查询失败，降级为本地 JSON 图谱")
        related = local_related_concepts(concepts)
        return f"你可能还想了解：{'、'.join(related)}。" if related else ""


def _run_neo4j(concepts: list[str]) -> str:
    driver = _load_graph()
    matched = []
    for concept in concepts:
        name = _match_name(driver, concept)
        if name and name not in matched:
            matched.append(name)

    candidate_groups = []
    for name in matched:
        candidate_groups.append([
            candidate for candidate in _find_related(driver, name)
            if candidate not in matched
        ])

    related = []
    max_group_size = max((len(group) for group in candidate_groups), default=0)
    for index in range(max_group_size):
        for group in candidate_groups:
            if index < len(group) and group[index] not in related:
                related.append(group[index])
                if len(related) >= 6:
                    break
        if len(related) >= 6:
            break

    if not related:
        return ""
    return f"你可能还想了解：{'、'.join(related)}。"
