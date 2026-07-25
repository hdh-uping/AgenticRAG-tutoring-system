"""graph_lookup Skill 的 Agent 调用入口。"""
from app.tools import graph_lookup


def run(concept: str) -> str:
    return graph_lookup(concept)
