"""独立推荐 Agent：由单独的 LangGraph 状态图驱动学习建议。"""
import json
import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.agent import _parse_action
from app.config import create_llm_client, get_settings
from app.skill_loader import execute_skill, load_all_skills, load_skill_instructions


SYSTEM_PROMPT = """你是《数据结构与算法》的独立学习推荐 Agent。

你的职责不是回答原问题，而是在教学 Agent 完成回答后，从知识图谱候选中选择最多 3 个真正相关的后续概念。

每轮只能输出一个 JSON 对象：

调用技能：
{{"reason_summary":"为什么需要候选概念","action":"related_concepts","input":"逗号分隔的当前概念"}}

完成推荐：
{{"reason_summary":"选择原则","action":"finish","recommendations":[{{"concept":"候选概念A","reason":"它与本轮问题的具体联系"}}]}}

规则：
- 有当前概念时，finish 前必须调用一次 related_concepts。
- 只能推荐 Observation 中出现的候选名称，不得创造 PREREQUISITE 或学习依赖。
- 推荐应结合原问题和教学回答，排除已经讲过的内容，最多选择 2 个。
- 比较多个概念的问题，应尽量让推荐覆盖比较双方，而不是只推荐一侧。
- reason 必须说明这个候选如何承接本轮问题，不能只说“有助于理解”。
- 候选都不合适时，recommendations 返回空数组。
- 不要回答原问题，不要输出 Markdown 或 JSON 之外的文字。"""


def _candidate_names(observation: str) -> list[str]:
    match = re.search(r"你可能还想了解[：:]\s*(.*?)。?\s*$", observation)
    if not match:
        return []
    return [item.strip() for item in re.split(r"[、,，]", match.group(1)) if item.strip()]


def _parse_recommendation_action(raw: str) -> tuple[str, str, str, list[dict]]:
    candidate = raw.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        payload = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        reason, action, argument = _parse_action(raw)
        return reason, action, argument, []
    if not isinstance(payload, dict):
        return "", "", "", []
    reason = str(payload.get("reason_summary") or "").strip()
    action = str(payload.get("action") or "").strip()
    argument = str(payload.get("input") or "").strip()
    recommendations = payload.get("recommendations")
    return reason, action, argument, recommendations if isinstance(recommendations, list) else []


def _validated_advice(proposed: list[dict], candidates: list[str]) -> tuple[str, list[str]]:
    items = []
    selected = []
    for item in proposed:
        if not isinstance(item, dict):
            continue
        concept = str(item.get("concept") or "").strip()
        reason = re.sub(r"\s+", " ", str(item.get("reason") or "")).strip(" ，。")
        if concept not in candidates or concept in selected or not reason:
            continue
        selected.append(concept)
        items.append(f"- **{concept}**：{reason[:120]}。")
        if len(items) >= 2:
            break
    if not items:
        return "", []
    return "## 推荐继续学习\n\n" + "\n".join(items), selected


def _fallback_advice(candidates: list[str], concepts: list[str]) -> str:
    if not candidates:
        return ""

    selected = []
    if len(concepts) > 1:
        # 比较题优先为每个概念各保留一个直接操作，避免又偏向其中一侧。
        for concept in concepts:
            direct = next(
                (candidate for candidate in candidates if candidate.startswith(f"{concept}·")),
                None,
            )
            if direct and direct not in selected:
                selected.append(direct)
            if len(selected) >= 2:
                break
    for candidate in candidates:
        if candidate not in selected:
            selected.append(candidate)
        if len(selected) >= 2:
            break

    comparison = "、".join(concepts[:2])
    items = []
    for candidate in selected:
        owner = next((concept for concept in concepts if candidate.startswith(f"{concept}·")), "")
        if len(concepts) > 1 and owner:
            reason = (
                f"把本轮对“{comparison}”的比较落实到“{candidate}”的具体步骤，"
                "可以继续观察两种结构的操作端和数据流向"
            )
        else:
            reason = f"它与本轮涉及的“{concepts[0]}”直接关联，可以把当前结论延伸到具体操作"
        items.append(f"- **{candidate}**：{reason}。")
    return "## 推荐继续学习\n\n" + "\n".join(items)


class RecommendationState(TypedDict, total=False):
    """推荐 Agent 子图的一次请求状态，拥有独立决策循环。"""

    concepts: list[str]
    question: str
    answer: str
    prefs: dict
    max_iter: int
    messages: list[dict]
    trace: list[dict]
    turn: int
    route: str
    pending_raw: str
    pending_reason: str
    pending_input: str
    candidates: list[str]
    skill_called: bool
    advice: str
    iterations: int


def build_recommendation_graph(skills: list[dict], client, settings):
    """构建独立推荐 Agent 的 LangGraph 状态图。"""
    related_meta = next(
        (skill for skill in skills if skill.get("name") == "related_concepts"),
        {},
    )
    related_description = str(related_meta.get("description") or "")

    def prepare(state: RecommendationState) -> dict:
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT + (
                    f"\n\n可用 Skill 元数据：\n- related_concepts: {related_description}"
                    if related_description else ""
                ),
            },
            {
                "role": "user",
                "content": (
                    f"原问题：{state.get('question', '')[:1000]}\n"
                    f"教学回答摘要：{state.get('answer', '')[:1800]}\n"
                    f"当前图谱概念：{', '.join(state['concepts'])}\n"
                    f"回答偏好：{state.get('prefs') or {}}"
                ),
            },
        ]
        return {
            "messages": messages,
            "trace": [],
            "turn": 0,
            "candidates": [],
            "skill_called": False,
            "advice": "",
            "route": "decide",
        }

    def decide(state: RecommendationState) -> dict:
        turn = state.get("turn", 0) + 1
        max_iter = state.get("max_iter", 2)
        messages = list(state["messages"])
        trace = list(state["trace"])
        try:
            response = client.chat.completions.create(
                model=settings.llm_model,
                messages=messages,
                temperature=0.1,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
        except Exception as exc:
            trace.append({
                "agent": "recommendation",
                "turn": turn,
                "action": "LLM_FALLBACK",
                "reason": type(exc).__name__,
            })
            return {"turn": turn, "trace": trace, "route": "fallback"}

        reason, action, argument, proposed = _parse_recommendation_action(raw)
        normalized_action = action.lower()
        if normalized_action == "related_concepts":
            if state.get("skill_called", False):
                trace.append({
                    "agent": "recommendation",
                    "turn": turn,
                    "action": "DUPLICATE_SKILL_BLOCKED",
                    "reason_summary": reason[:200],
                })
                messages.extend([
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": "候选已经取得，请直接 finish。"},
                ])
                return {
                    "turn": turn,
                    "trace": trace,
                    "messages": messages,
                    "route": "decide" if turn < max_iter else "fallback",
                }
            return {
                "turn": turn,
                "pending_raw": raw,
                "pending_reason": reason,
                "pending_input": argument,
                "route": "related_concepts",
            }

        if action.upper() in ("FINISH", "FINAL_ANSWER"):
            if not state.get("skill_called", False):
                trace.append({
                    "agent": "recommendation",
                    "turn": turn,
                    "action": "FINISH_REJECTED_NO_SKILL",
                    "reason_summary": reason[:200],
                })
                messages.extend([
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": "请先调用 related_concepts，再决定是否推荐。"},
                ])
                return {
                    "turn": turn,
                    "trace": trace,
                    "messages": messages,
                    "route": "decide" if turn < max_iter else "fallback",
                }
            advice, selected = _validated_advice(proposed, state.get("candidates", []))
            trace.append({
                "agent": "recommendation",
                "turn": turn,
                "action": "finish",
                "reason_summary": reason[:200],
                "selected": selected,
            })
            return {
                "turn": turn,
                "trace": trace,
                "advice": advice,
                "iterations": turn,
                "route": "finalize",
            }

        trace.append({
            "agent": "recommendation",
            "turn": turn,
            "action": "FORMAT_ERROR",
            "output_preview": raw[:200],
        })
        messages.extend([
            {"role": "assistant", "content": raw},
            {"role": "user", "content": "格式无效，请只输出规定的 JSON 动作。"},
        ])
        return {
            "turn": turn,
            "trace": trace,
            "messages": messages,
            "route": "decide" if turn < max_iter else "fallback",
        }

    def related_concepts(state: RecommendationState) -> dict:
        skill_input = state.get("pending_input") or ", ".join(state["concepts"])
        observation = execute_skill("related_concepts", skill_input, skills)
        instructions = load_skill_instructions("related_concepts", skills)
        candidates = _candidate_names(observation)
        messages = list(state["messages"])
        messages.extend([
            {"role": "assistant", "content": state.get("pending_raw", "")},
            {
                "role": "user",
                "content": (
                    f'<skill_instructions name="related_concepts">\n{instructions}\n'
                    f"</skill_instructions>\n\nObservation:\n{observation}"
                    if instructions else f"Observation:\n{observation}"
                ),
            },
        ])
        trace = list(state["trace"])
        trace.append({
            "agent": "recommendation",
            "turn": state["turn"],
            "action": "related_concepts",
            "reason_summary": state.get("pending_reason", "")[:200],
            "input": skill_input[:200],
            "candidates": candidates,
            "skill_instructions_loaded": bool(instructions),
        })
        if not candidates:
            return {
                "messages": messages,
                "trace": trace,
                "candidates": [],
                "skill_called": True,
                "advice": "",
                "iterations": state["turn"],
                "route": "finalize",
            }
        return {
            "messages": messages,
            "trace": trace,
            "candidates": candidates,
            "skill_called": True,
            "route": "decide" if state["turn"] < state.get("max_iter", 2) else "fallback",
        }

    def fallback(state: RecommendationState) -> dict:
        candidates = list(state.get("candidates", []))
        trace = list(state["trace"])
        if not state.get("skill_called", False):
            observation = execute_skill(
                "related_concepts", ", ".join(state["concepts"]), skills
            )
            instructions = load_skill_instructions("related_concepts", skills)
            candidates = _candidate_names(observation)
            trace.append({
                "agent": "recommendation",
                "turn": len(trace) + 1,
                "action": "FALLBACK_RELATED_CONCEPTS",
                "candidates": candidates,
                "skill_instructions_loaded": bool(instructions),
            })
        return {
            "candidates": candidates,
            "trace": trace,
            "advice": _fallback_advice(candidates, state["concepts"]),
            "iterations": min(state.get("max_iter", 2), len(trace)),
            "route": "finalize",
        }

    def finalize(state: RecommendationState) -> dict:
        return {
            "advice": state.get("advice", ""),
            "iterations": state.get("iterations", state.get("turn", 0)),
        }

    graph = StateGraph(RecommendationState)
    graph.add_node("prepare", prepare)
    graph.add_node("decide", decide)
    graph.add_node("related_concepts", related_concepts)
    graph.add_node("fallback", fallback)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "decide")
    graph.add_conditional_edges(
        "decide",
        lambda state: state["route"],
        {
            "decide": "decide",
            "related_concepts": "related_concepts",
            "fallback": "fallback",
            "finalize": "finalize",
        },
    )
    graph.add_conditional_edges(
        "related_concepts",
        lambda state: state["route"],
        {"decide": "decide", "fallback": "fallback", "finalize": "finalize"},
    )
    graph.add_edge("fallback", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def run_recommendation_agent(
    concepts_involved: list[str] | None = None,
    *,
    question: str = "",
    answer: str = "",
    prefs: dict | None = None,
    max_iter: int = 2,
) -> dict:
    """运行与教学 Agent 相互独立的推荐状态图。"""
    concepts = [item.strip() for item in (concepts_involved or []) if item.strip()]
    if not concepts:
        return {"advice": "", "trace": [], "iterations": 0}

    skills = load_all_skills()
    settings = get_settings()
    try:
        client = create_llm_client()
        graph = build_recommendation_graph(skills, client, settings)
        state = graph.invoke({
            "concepts": concepts,
            "question": question,
            "answer": answer,
            "prefs": prefs or {},
            "max_iter": max_iter,
        })
        return {
            "advice": state["advice"],
            "trace": state["trace"],
            "iterations": state["iterations"],
        }
    except Exception as exc:
        trace = [{
            "agent": "recommendation",
            "turn": 1,
            "action": "LLM_FALLBACK",
            "reason": type(exc).__name__,
        }]
        observation = execute_skill("related_concepts", ", ".join(concepts), skills)
        instructions = load_skill_instructions("related_concepts", skills)
        candidates = _candidate_names(observation)
        trace.append({
            "agent": "recommendation",
            "turn": len(trace) + 1,
            "action": "FALLBACK_RELATED_CONCEPTS",
            "candidates": candidates,
            "skill_instructions_loaded": bool(instructions),
        })
        return {
            "advice": _fallback_advice(candidates, concepts),
            "trace": trace,
            "iterations": min(max_iter, len(trace)),
        }
