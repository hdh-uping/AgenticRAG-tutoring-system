"""双 Agent 总工作流：父 LangGraph 编排教学与推荐两个独立 Agent 子图。"""
import logging
from collections.abc import Callable
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.agent import run_teaching_agent
from app.recommendation_agent import run_recommendation_agent


logger = logging.getLogger(__name__)


class TutoringWorkflowState(TypedDict, total=False):
    """父图状态；两个 Agent 的内部消息和路由状态封装在各自子图中。"""

    question: str
    session_history: list[dict]
    prefs: dict
    teaching_max_iter: int
    recommendation_max_iter: int
    teaching_result: dict
    recommendation_result: dict
    recommendation_error: str
    route: str
    answer: str
    recommendation: str
    trace: list[dict]
    recommendation_trace: list[dict]
    iterations: int
    concepts_involved: list[str]
    sources: list[dict]


def build_tutoring_workflow(
    teaching_runner: Callable[..., dict] | None = None,
    recommendation_runner: Callable[..., dict] | None = None,
):
    """构建父图；注入 runner 便于隔离测试，默认调用两个 Agent 子图。"""
    teaching_runner = teaching_runner or run_teaching_agent
    recommendation_runner = recommendation_runner or run_recommendation_agent

    def prepare_context(state: TutoringWorkflowState) -> dict:
        return {
            "session_history": state.get("session_history") or [],
            "prefs": state.get("prefs") or {},
            "teaching_max_iter": state.get("teaching_max_iter", 3),
            "recommendation_max_iter": state.get("recommendation_max_iter", 2),
            "recommendation_result": {
                "advice": "",
                "trace": [],
                "iterations": 0,
            },
            "recommendation_error": "",
        }

    def teaching_agent(state: TutoringWorkflowState) -> dict:
        result = teaching_runner(
            question=state["question"],
            max_iter=state.get("teaching_max_iter", 3),
            session_history=state.get("session_history") or [],
            prefs=state.get("prefs") or {},
        )
        return {
            "teaching_result": result,
            "route": (
                "recommendation_agent"
                if result.get("concepts_involved")
                else "assemble_response"
            ),
        }

    def recommendation_agent(state: TutoringWorkflowState) -> dict:
        teaching = state["teaching_result"]
        try:
            result = recommendation_runner(
                teaching.get("concepts_involved") or [],
                question=state["question"],
                answer=teaching.get("answer", ""),
                prefs=state.get("prefs") or {},
                max_iter=state.get("recommendation_max_iter", 2),
            )
            return {"recommendation_result": result, "route": "assemble_response"}
        except Exception as exc:
            logger.exception("推荐 Agent 子图执行失败，父图将保留教学答案")
            return {
                "recommendation_error": type(exc).__name__,
                "route": "recommendation_fallback",
            }

    def recommendation_fallback(state: TutoringWorkflowState) -> dict:
        return {
            "recommendation_result": {
                "advice": "",
                "trace": [{
                    "agent": "recommendation",
                    "turn": 0,
                    "action": "PARENT_GRAPH_FALLBACK",
                    "reason": state.get("recommendation_error", "UnknownError"),
                }],
                "iterations": 0,
            }
        }

    def assemble_response(state: TutoringWorkflowState) -> dict:
        teaching = state["teaching_result"]
        recommendation_result = state.get("recommendation_result") or {}
        recommendation = str(recommendation_result.get("advice") or "")
        answer = str(teaching.get("answer") or "")
        if recommendation:
            answer += f"\n\n{recommendation}"
        return {
            "answer": answer,
            "recommendation": recommendation,
            "trace": teaching.get("trace") or [],
            "recommendation_trace": recommendation_result.get("trace") or [],
            "iterations": int(teaching.get("iterations") or 0),
            "concepts_involved": teaching.get("concepts_involved") or [],
            "sources": teaching.get("sources") or [],
        }

    graph = StateGraph(TutoringWorkflowState)
    graph.add_node("prepare_context", prepare_context)
    graph.add_node("teaching_agent", teaching_agent)
    graph.add_node("recommendation_agent", recommendation_agent)
    graph.add_node("recommendation_fallback", recommendation_fallback)
    graph.add_node("assemble_response", assemble_response)
    graph.add_edge(START, "prepare_context")
    graph.add_edge("prepare_context", "teaching_agent")
    graph.add_conditional_edges(
        "teaching_agent",
        lambda state: state["route"],
        {
            "recommendation_agent": "recommendation_agent",
            "assemble_response": "assemble_response",
        },
    )
    graph.add_conditional_edges(
        "recommendation_agent",
        lambda state: state["route"],
        {
            "recommendation_fallback": "recommendation_fallback",
            "assemble_response": "assemble_response",
        },
    )
    graph.add_edge("recommendation_fallback", "assemble_response")
    graph.add_edge("assemble_response", END)
    return graph.compile()


def run_tutoring_workflow(
    question: str,
    *,
    session_history: list[dict] | None = None,
    prefs: dict | None = None,
    teaching_max_iter: int = 3,
    recommendation_max_iter: int = 2,
) -> dict:
    """通过一张父图执行完整双 Agent 工作流。"""
    graph = build_tutoring_workflow()
    state = graph.invoke({
        "question": question,
        "session_history": session_history or [],
        "prefs": prefs or {},
        "teaching_max_iter": teaching_max_iter,
        "recommendation_max_iter": recommendation_max_iter,
    })
    return {
        "answer": state["answer"],
        "recommendation": state["recommendation"],
        "trace": state["trace"],
        "recommendation_trace": state["recommendation_trace"],
        "iterations": state["iterations"],
        "concepts_involved": state["concepts_involved"],
        "sources": state["sources"],
    }
