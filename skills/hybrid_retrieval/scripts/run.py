"""hybrid_retrieval Skill 的 Agent 调用入口。"""
from app.tools import hybrid_retrieval


def run(query: str, top_k: int = 5) -> str:
    return hybrid_retrieval(query, top_k=top_k)
