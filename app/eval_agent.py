"""旧入口兼容层；当前职责是关联推荐，不执行学生评估。"""
from app.recommendation_agent import run_recommendation_agent


def run_eval_agent(
    question: str = "",
    concepts_involved: list[str] = None,
) -> dict:
    return run_recommendation_agent(concepts_involved, question=question)
