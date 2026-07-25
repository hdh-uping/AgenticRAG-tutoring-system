"""
教学 Agent — LangGraph 编排的 Plan-guided ReAct 状态图。

核心逻辑：
  Agent 只接收 Skill 的 name/description → 按描述选择 Skill
  每轮：decide 节点输出结构化动作 → execute_skill 节点执行 → 回到 decide
  终止：证据满足计划要求，或满 3 轮后进入 fallback，再生成并验证答案
"""
import json
import re
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.config import create_llm_client, get_settings
from app.evidence import EvidencePool, append_references
from app.skill_loader import (
    build_skill_prompt,
    execute_skill,
    load_all_skills,
    load_skill_instructions,
)

# ── System Prompt 模板 ─────────────────────────────────────────

SYSTEM_PROMPT = """你是《数据结构与算法》的课程助教。学生问你问题，你可以通过调用技能来查资料、回答问题。

{skill_descriptions}

## 工作方式

每轮你**必须**只输出一个 JSON 对象，不要输出 Markdown 代码块或额外文字：

调用技能时：
{{"reason_summary":"一句话说明需要什么证据","action":"技能名","input":"参数"}}

完成回答时：
{{"reason_summary":"证据为什么已经足够","action":"finish"}}

action 只能是 {allowed_actions}。

## 当前任务计划

{task_plan}

只有计划中的完成条件均已满足时才能 finish；Observation 缺少某项时继续调用合适的技能。

重要规则：
- 根据上方每个 Skill 的 description 选择最适合当前证据目标的 Skill，不要按固定顺序调用。
- **拿到结果后首要任务：判断核心问题是否已能回答。** 能答就立刻 FINISH。不要因为「还可以更完美」「还能展开一下」而拖延。
- rerank_score 只表示内容相关度，不等于证据已经覆盖了全部问题要求。
- 只有当检索结果明显缺了核心内容（比如问了代码但只拿到概念解释），才再搜一轮。最多 1 轮补充。
- 如果查不到相关内容，诚实告知。

## 用户偏好

{prefs_text}

根据偏好调整回答方式。

## 证据约束

{evidence_rule}"""

# ── 工具调用解析 ──────────────────────────────────────────────

def _parse_action(raw: str) -> tuple[str, str, str]:
    """解析结构化动作，并兼容旧版 Thought/Action 文本格式。"""
    thought = ""
    action_name = ""
    action_arg = ""

    candidate = raw.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)

    try:
        payload = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        payload = None

    if isinstance(payload, dict):
        thought = str(payload.get("reason_summary") or payload.get("thought") or "").strip()
        action_name = str(payload.get("action") or "").strip()
        if action_name.lower() in ("finish", "final_answer"):
            action_name = "FINISH"
            action_arg = str(payload.get("answer") or payload.get("input") or "").strip()
        else:
            action_arg = str(payload.get("input") or payload.get("argument") or "").strip()
        return thought, action_name, action_arg

    thought_match = re.search(r"(?im)^Thought\s*:\s*(.*)$", candidate)
    if thought_match:
        thought = thought_match.group(1).strip()

    action_match = re.search(r"(?im)^Action\s*:\s*", candidate)
    if not action_match:
        return thought, action_name, action_arg
    action_str = candidate[action_match.end():].strip()

    # FINISH[xxx] — 在完整原文上匹配，保留多行答案。
    m = re.match(r"FINISH\s*\[(.*)\]\s*$", action_str, re.DOTALL | re.IGNORECASE)
    if not m:
        m = re.match(r"FINISH\s*\[(.*)$", action_str, re.DOTALL | re.IGNORECASE)
    if m:
        return thought, "FINISH", m.group(1).strip()

    m = re.match(r"(\w+)\[(.*)\]\s*$", action_str, re.DOTALL)
    if m:
        action_name = m.group(1)
        action_arg = m.group(2).strip()

    return thought, action_name, action_arg


def _execute_skill(name: str, arg: str, skills: list[dict]) -> str:
    """通过 SkillLoader 执行 Skill 并返回文本结果。"""
    if name.upper() in ("FINISH", "FINAL_ANSWER"):
        return arg  # 这是最终答案，不执行
    return execute_skill(name, arg, skills)


# ── 硬性终止检查 ──────────────────────────────────────────────

def _should_force_stop(observation: str, iteration: int, max_iter: int = 3) -> tuple[bool, str]:
    """只以轮次上限终止；相关度分数不能证明证据充分。"""
    del observation
    if iteration >= max_iter:
        return True, f"已达 {max_iter} 轮上限"
    return False, ""


def _requires_evidence(question: str, session_history: list[dict] | None = None) -> bool:
    """知识性问题必须检索；寒暄和基于历史的表达调整可以直接回答。"""
    normalized = re.sub(r"[\s，。！？!?]", "", question).lower()
    conversational = {
        "你好", "您好", "谢谢", "感谢", "再见", "好的", "明白了", "知道了",
    }
    if normalized in conversational:
        return False

    if session_history:
        follow_up_markers = (
            "再简单", "再详细", "简短一点", "详细一点", "换种说法", "继续",
            "没看懂", "上面", "刚才", "举个例子", "重新解释",
        )
        if any(marker in normalized for marker in follow_up_markers):
            return False
    return True


def _call_key(action_name: str, action_arg: str) -> tuple[str, str]:
    return action_name.lower(), re.sub(r"\s+", " ", action_arg).strip().lower()


def _build_task_plan(question: str) -> dict:
    """把回答要求显式化，但不把任务硬绑定到具体 Skill。"""
    normalized = re.sub(r"\s+", "", question).lower()
    declines_code = any(marker in normalized for marker in (
        "不需要代码", "不要代码", "无需代码", "只讲思路", "只说思路",
    ))
    wants_code = (
        any(marker in normalized for marker in ("代码", "编写", "实现", "算法"))
        and not declines_code
    )
    wants_complexity = "复杂度" in normalized
    wants_comparison = any(marker in normalized for marker in ("区别", "比较", "对比", "不同"))
    wants_calculation = any(marker in normalized for marker in ("计算", "地址", "求出", "是多少"))
    wants_example = any(marker in normalized for marker in ("举例", "例子", "示例"))
    wants_background = any(marker in normalized for marker in ("先补充", "先解释", "概念"))

    requirements = []
    checks = []
    if wants_background:
        requirements.append("先补充完成题目所需的核心概念")
    if wants_comparison:
        requirements.extend(["覆盖比较双方", "说明至少两个核心差异或操作规则"])
        checks.append("comparison")
    elif wants_calculation:
        requirements.extend(["给出适用公式或规则", "展示代入或推导过程", "给出明确结论"])
        checks.append("calculation")
    else:
        requirements.append("直接回应问题的核心目标")

    if wants_code:
        requirements.extend([
            "解释实现思路和关键数据结构",
            "给出包含函数定义与核心流程的完整代码",
            "说明关键边界条件或失败处理",
        ])
        checks.extend(["function_code", "edge_cases"])
    if wants_complexity:
        requirements.append("给出时间复杂度或空间复杂度结论")
        checks.append("complexity")
    if wants_example:
        requirements.append("提供与问题直接相关的示例")
        checks.append("example")

    if wants_code:
        steps = [
            {"goal": "获取能支持实现思路、完整函数和边界处理的可靠证据"},
            {"goal": "首轮证据不完整时，按 Skill 描述选择其他来源补充"},
        ]
    else:
        steps = [{"goal": "获取能直接支持问题结论的可靠证据"}]

    question_type = (
        "implementation" if wants_code else
        "comparison" if wants_comparison else
        "calculation" if wants_calculation else
        "knowledge"
    )
    return {
        "question_type": question_type,
        "requirements": list(dict.fromkeys(requirements)),
        "steps": steps,
        "checks": list(dict.fromkeys(checks)),
    }


def _plan_summary(plan: dict) -> str:
    requirements = "；".join(plan.get("requirements", []))
    return f"{plan.get('question_type', 'knowledge')}：{requirements}"[:300]


def _fallback_input(skill_name: str, question: str) -> str:
    if skill_name != "graph_lookup":
        return question
    concepts = _extract_concepts([], question)
    return concepts[0] if concepts else question


def _available_agent_skills(skills: list[dict]) -> list[str]:
    """返回当前教学 Agent 可见的 Skill 名，作为动作白名单。"""
    return [
        str(skill.get("name")) for skill in skills
        if skill.get("name") and skill.get("in_agent_loop", True)
    ]


def _fallback_skill(available_skills: list[str]) -> str:
    """模型未取得证据时使用通用检索兜底，不参与正常路由。"""
    if "hybrid_retrieval" in available_skills:
        return "hybrid_retrieval"
    return available_skills[0] if available_skills else ""


def _generate_final_answer(
    client,
    settings,
    messages: list[dict],
    task_plan: dict,
    verification_feedback: list[str] | None = None,
) -> str:
    """在动作循环之外生成正文，避免把多行答案嵌入动作 JSON。"""
    final_messages = [*messages, {
        "role": "user",
        "content": (
            "证据收集已经结束。现在只输出面向学生的最终 Markdown 回答，不要输出 JSON、"
            "reason_summary、action、Thought 或 Observation。回答必须基于已有 Observation；"
            "引用教材时在相关句子后标注 chunk id 和页码，但不要自行生成“参考来源”章节。"
            "不要给算法附加证据中没有的性质；涉及相等元素时要明确插入位置策略，"
            "不能把插在已有相等元素之前误称为稳定。\n"
            "不要主动延伸到 Observation 中未出现的新术语、结构或操作。\n"
            f"必须覆盖这些任务要求：{json.dumps(task_plan.get('requirements', []), ensure_ascii=False)}。\n"
            + (
                "上一次验证发现以下问题，必须逐项修正："
                + "；".join(verification_feedback or [])
                if verification_feedback else ""
            )
        ),
    }]
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=final_messages,
        temperature=0.1,
        max_tokens=2048,
    )
    return _clean_answer(response.choices[0].message.content.strip())


def _deterministic_answer_issues(plan: dict, answer: str, evidence: EvidencePool) -> list[str]:
    """无需 LLM 的可验证约束，防止格式、代码和引用层面的明显遗漏。"""
    issues = []
    stripped = answer.lstrip()
    if stripped.startswith("{") and '"action"' in stripped[:300]:
        issues.append("最终回答泄漏了内部 JSON 动作协议")

    checks = set(plan.get("checks", []))
    code_blocks = re.findall(r"```(?:c|cpp|c\+\+)?\s*\n(.*?)```", answer, re.DOTALL | re.IGNORECASE)
    if "function_code" in checks:
        function_pattern = re.compile(
            r"\b(?:int|void|bool|long|float|double|[A-Za-z_]\w*\s*\*?)\s+"
            r"[A-Za-z_]\w*\s*\([^;{}]*\)\s*\{",
            re.DOTALL,
        )
        if not any(function_pattern.search(block) for block in code_blocks):
            issues.append("代码题缺少包含函数定义和核心流程的完整实现")
    if "edge_cases" in checks and not re.search(
        r"边界|失败|为空|空表|越界|溢出|容量|MaxSize|return\s+-?\d|if\s*\(", answer
    ):
        issues.append("代码题没有说明边界条件或失败处理")
    if "complexity" in checks and not re.search(r"O\s*\([^)]*\)", answer):
        issues.append("缺少复杂度结论")
    if "calculation" in checks and not any(token in answer for token in ("=", "因此", "所以")):
        issues.append("计算题缺少推导过程或明确结论")

    known_ids = {source["id"] for source in evidence.to_sources()}
    cited_ids = set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+_chunk_\d+", answer))
    unknown_ids = sorted(cited_ids - known_ids)
    if unknown_ids:
        issues.append("回答引用了未检索到的来源：" + "、".join(unknown_ids))
    return issues


def _verify_answer(
    client,
    settings,
    question: str,
    task_plan: dict,
    answer: str,
    evidence: EvidencePool,
    observations: list[dict],
    allowed_skills: list[str] | None = None,
) -> dict:
    """教学 Agent 的证据审查阶段；它是教学 Agent 内部步骤，不是第三个 Agent。"""
    allowed_skills = allowed_skills or ["hybrid_retrieval", "graph_lookup"]
    allowed_skill_set = set(allowed_skills)
    fallback_skill = _fallback_skill(allowed_skills)
    deterministic_issues = _deterministic_answer_issues(task_plan, answer, evidence)
    evidence_text = "\n\n".join(
        f"[{item['skill']} | input={item['input']}]\n{item['text']}"
        for item in observations
    )[-7000:]
    verifier_messages = [
        {
            "role": "system",
            "content": (
                "你是教学 Agent 内部的答案验证阶段，不是新的 Agent。只输出 JSON。"
                "检查草稿是否覆盖任务计划、每个关键结论是否得到证据支持、代码是否真正实现用户要求。"
                "不得因为语言不够漂亮而判失败。status 只能是 pass、revise、retrieve_more。"
                "证据充分但表述或代码细节需修正时用 revise，只列出问题，不要重写整篇答案；"
                "缺少完成答案所必需的资料时用 retrieve_more，并给 suggested_skill 和 suggested_query。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps({
                "question": question,
                "task_plan": task_plan,
                "deterministic_issues": deterministic_issues,
                "draft_answer": answer,
                "evidence": evidence_text,
                "allowed_skills": allowed_skills,
                "output_schema": {
                    "status": "pass|revise|retrieve_more",
                    "issues": ["问题"],
                    "missing_requirements": ["未覆盖要求"],
                    "suggested_skill": "从 allowed_skills 选择或使用空字符串",
                    "suggested_query": "补检索查询或空字符串",
                },
            }, ensure_ascii=False),
        },
    ]
    try:
        raw = ""
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=verifier_messages,
            temperature=0.0,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
        payload = json.loads(raw)
    except Exception as exc:
        return {
            "status": "retrieve_more" if deterministic_issues else "pass",
            "issues": deterministic_issues,
            "missing_requirements": deterministic_issues,
            "suggested_skill": fallback_skill if deterministic_issues else "",
            "suggested_query": question,
            "revised_answer": "",
            "fallback_reason": type(exc).__name__,
            "output_preview": raw[:200] if raw else "",
        }

    status = str(payload.get("status") or "").strip().lower()
    if status not in {"pass", "revise", "retrieve_more"}:
        status = "retrieve_more" if deterministic_issues else "pass"
    issues = [str(item)[:200] for item in payload.get("issues", []) if str(item).strip()]
    missing = [
        str(item)[:200] for item in payload.get("missing_requirements", [])
        if str(item).strip()
    ]
    for issue in deterministic_issues:
        if issue not in issues:
            issues.append(issue)
        if issue not in missing:
            missing.append(issue)
    if deterministic_issues and status == "pass":
        status = "retrieve_more"
    suggested_skill = str(payload.get("suggested_skill") or "").strip()
    if suggested_skill not in allowed_skill_set:
        suggested_skill = fallback_skill if status == "retrieve_more" else ""
    return {
        "status": status,
        "issues": issues,
        "missing_requirements": missing,
        "suggested_skill": suggested_skill,
        "suggested_query": str(payload.get("suggested_query") or "").strip()[:500],
        "revised_answer": "",
    }


# ═══════════════════════════════════════════════════════════════
# LangGraph 状态图
# ═══════════════════════════════════════════════════════════════


class TeachingState(TypedDict, total=False):
    """教学图的一次请求状态；会话长期历史仍由业务 SQLite 管理。"""

    question: str
    max_iter: int
    session_history: list[dict]
    prefs: dict
    task_plan: dict
    requires_evidence: bool
    available_skills: list[str]
    messages: list[dict]
    trace: list[dict]
    turn: int
    observations: list[dict]
    seen_calls: list[tuple[str, str]]
    disclosed_skills: list[str]
    route: str
    pending_raw: str
    pending_action: str
    pending_input: str
    pending_answer: str
    final_answer: str
    force_stopped: bool
    force_reason: str
    verification: dict
    sources: list[dict]
    concepts_involved: list[str]
    iterations: int


def _evidence_from_observations(observations: list[dict]) -> EvidencePool:
    """由图状态恢复证据池，避免把运行时对象写入 LangGraph State。"""
    evidence = EvidencePool()
    for item in observations:
        skill_name = str(item.get("skill") or "")
        text = str(item.get("text") or "")
        if skill_name and text:
            evidence.add_observation(skill_name, text)
    return evidence


def build_teaching_graph(skills: list[dict], client, settings):
    """构建教学 Agent 状态图；运行时依赖通过闭包注入，不进入图状态。"""
    skill_descriptions = build_skill_prompt(skills)

    def observation_message(
        skill_name: str,
        observation: str,
        disclosed_skills: list[str],
    ) -> tuple[str, bool, list[str]]:
        """首次触发 Skill 时才渐进式披露完整说明。"""
        disclosed = list(disclosed_skills)
        instructions = ""
        loaded_now = False
        if skill_name and skill_name not in disclosed:
            instructions = load_skill_instructions(skill_name, skills)
            disclosed.append(skill_name)
            loaded_now = bool(instructions)
        if not instructions:
            return f"Observation:\n{observation}", loaded_now, disclosed
        return (
            f'<skill_instructions name="{skill_name}">\n{instructions}\n'
            f"</skill_instructions>\n\nObservation:\n{observation}",
            loaded_now,
            disclosed,
        )

    def prepare(state: TeachingState) -> dict:
        question = state["question"]
        session_history = state.get("session_history") or []
        prefs = state.get("prefs") or {}

        depth_map = {
            "beginner": "学生是初学者，请用类比、从基础讲起，多铺垫",
            "intermediate": "按正常节奏讲解",
            "advanced": "简洁直接，直击核心，不用铺垫",
        }
        code_map = {"full": "给出完整代码", "idea": "只讲思路，不写完整代码"}
        style_map = {"casual": "语气口语化、亲切", "academic": "语气正式、学术化"}
        length_map = {
            "concise": "回答简洁，只保留结论和必要理由",
            "balanced": "回答长度适中",
            "detailed": "回答详细，补充步骤、原因和必要示例",
        }
        prefs_text = "\n".join([
            f"- 讲解深度：{depth_map.get(prefs.get('depth', 'intermediate'), '')}",
            f"- 代码展示：{code_map.get(prefs.get('show_code', 'full'), '')}",
            f"- 语气风格：{style_map.get(prefs.get('style', 'casual'), '')}",
            f"- 回答长度：{length_map.get(prefs.get('response_length', 'balanced'), '')}",
        ])

        requires_evidence = _requires_evidence(question, session_history)
        task_plan = _build_task_plan(question)
        if not requires_evidence:
            task_plan = {
                "question_type": "conversation",
                "requirements": ["直接回应用户当前的会话请求"],
                "steps": [],
                "checks": [],
            }
        available_skills = _available_agent_skills(skills)
        evidence_rule = (
            "这是知识性问题。至少成功调用一次可用 Skill 并取得有效教材或图谱证据后才能 finish。"
            if requires_evidence else
            "这是寒暄或基于历史的表达调整，可以直接 finish。"
        )
        system_prompt = SYSTEM_PROMPT.format(
            skill_descriptions=skill_descriptions,
            prefs_text=prefs_text,
            evidence_rule=evidence_rule,
            allowed_actions="、".join([*available_skills, "finish"]),
            task_plan=json.dumps(task_plan, ensure_ascii=False, indent=2),
        )
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(session_history)
        messages.append({"role": "user", "content": f"<question>{question}</question>"})
        return {
            "task_plan": task_plan,
            "requires_evidence": requires_evidence,
            "available_skills": available_skills,
            "messages": messages,
            "trace": [{
                "turn": 0,
                "action": "PLAN",
                "reason_summary": _plan_summary(task_plan),
                "plan": task_plan,
            }],
            "turn": 0,
            "observations": [],
            "seen_calls": [],
            "disclosed_skills": [],
            "final_answer": "",
            "force_stopped": False,
            "force_reason": "",
            "route": "decide",
        }

    def decide(state: TeachingState) -> dict:
        turn = state.get("turn", 0) + 1
        max_iter = state.get("max_iter", 3)
        messages = list(state["messages"])
        trace = list(state["trace"])
        available_skills = state["available_skills"]
        evidence = _evidence_from_observations(state.get("observations", []))

        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            temperature=0.1,
            max_tokens=2048,
        )
        raw = response.choices[0].message.content.strip()
        reason_summary, action_name, action_arg = _parse_action(raw)
        allowed_actions = {name.lower() for name in available_skills}
        allowed_actions.update({"finish", "final_answer"})
        if action_name and action_name.lower() not in allowed_actions:
            action_name = ""

        if not action_name:
            trace.append({
                "turn": turn,
                "action": "FORMAT_ERROR",
                "output_preview": raw[:200],
            })
            messages.extend([
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        "输出格式无效。请只返回 JSON："
                        '{"reason_summary":"...","action":"'
                        + "|".join([*available_skills, "finish"])
                        + '","input":"..."}；finish 时只需 action 和 reason_summary。'
                    ),
                },
            ])
            return {
                "turn": turn,
                "messages": messages,
                "trace": trace,
                "route": "decide" if turn < max_iter else "fallback",
            }

        action_trace = {
            "turn": turn,
            "reason_summary": reason_summary[:200],
            "action": action_name,
            "input": action_arg[:200]
            if action_name.upper() not in ("FINISH", "FINAL_ANSWER") else "",
        }
        trace.append(action_trace)

        if action_name.upper() in ("FINISH", "FINAL_ANSWER"):
            if state["requires_evidence"] and not evidence.has_evidence:
                trace[-1]["result"] = "finish_rejected_no_evidence"
                messages.extend([
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": "当前还没有有效教材或图谱证据。请先调用一个技能，不要直接 finish。",
                    },
                ])
                return {
                    "turn": turn,
                    "messages": messages,
                    "trace": trace,
                    "route": "decide" if turn < max_iter else "fallback",
                }
            trace[-1]["result"] = "agent_decided_finish"
            messages.append({"role": "assistant", "content": raw})
            return {
                "turn": turn,
                "messages": messages,
                "trace": trace,
                "pending_answer": action_arg,
                "route": "draft",
            }

        return {
            "turn": turn,
            "trace": trace,
            "pending_raw": raw,
            "pending_action": action_name,
            "pending_input": action_arg,
            "route": "execute_skill",
        }

    def execute_skill_node(state: TeachingState) -> dict:
        action_name = state["pending_action"]
        action_arg = state.get("pending_input", "")
        raw = state.get("pending_raw", "")
        turn = state["turn"]
        max_iter = state.get("max_iter", 3)
        trace = list(state["trace"])
        messages = list(state["messages"])
        seen_calls = list(state.get("seen_calls", []))
        call_key = _call_key(action_name, action_arg)

        if call_key in seen_calls:
            observation = "(已阻止重复调用：相同技能和参数已经执行过，请改写查询或直接作答。)"
            trace[-1]["result"] = "duplicate_call_blocked"
            trace[-1]["observation_preview"] = observation
            messages.extend([
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"Observation:\n{observation}"},
            ])
            return {
                "messages": messages,
                "trace": trace,
                "route": "decide" if turn < max_iter else "fallback",
            }

        seen_calls.append(call_key)
        observation = _execute_skill(action_name, action_arg, skills)
        observations = list(state.get("observations", []))
        evidence = _evidence_from_observations(observations)
        observations.append({"skill": action_name, "input": action_arg, "text": observation})
        added_sources = evidence.add_observation(action_name, observation)
        evidence_ids = [source["id"] for source in added_sources]
        rerank_scores = re.findall(r"rerank_score=(\d+\.?\d*)", observation)
        trace[-1]["observation_preview"] = observation[:200]
        if evidence_ids:
            trace[-1]["evidence_ids"] = evidence_ids
        if rerank_scores:
            trace[-1]["rerank_scores"] = [float(score) for score in rerank_scores]

        observation_content, loaded_now, disclosed = observation_message(
            action_name,
            observation,
            state.get("disclosed_skills", []),
        )
        trace[-1]["skill_instructions_loaded"] = loaded_now
        messages.extend([
            {"role": "assistant", "content": raw},
            {"role": "user", "content": observation_content},
        ])
        should_stop, reason = _should_force_stop(observation, turn, max_iter)
        return {
            "messages": messages,
            "trace": trace,
            "observations": observations,
            "seen_calls": seen_calls,
            "disclosed_skills": disclosed,
            "force_stopped": should_stop,
            "force_reason": reason if should_stop else "",
            "route": "fallback" if should_stop else "decide",
        }

    def draft(state: TeachingState) -> dict:
        answer = state.get("pending_answer", "")
        if not answer:
            answer = _generate_final_answer(
                client,
                settings,
                state["messages"],
                state["task_plan"],
            )
        return {"final_answer": answer, "route": "verify"}

    def fallback(state: TeachingState) -> dict:
        observations = list(state.get("observations", []))
        messages = list(state["messages"])
        trace = list(state["trace"])
        disclosed = list(state.get("disclosed_skills", []))
        evidence = _evidence_from_observations(observations)
        turn = state.get("turn", 0)

        if state["requires_evidence"] and not evidence.has_evidence:
            fallback_name = _fallback_skill(state["available_skills"])
            fallback_arg = _fallback_input(fallback_name, state["question"])
            observation = _execute_skill(fallback_name, fallback_arg, skills) if fallback_name else ""
            observations.append({
                "skill": fallback_name,
                "input": fallback_arg,
                "text": observation,
            })
            added_sources = evidence.add_observation(fallback_name, observation) if fallback_name else []
            observation_content, loaded_now, disclosed = observation_message(
                fallback_name,
                observation,
                disclosed,
            )
            trace.append({
                "turn": turn + 1,
                "action": "FALLBACK_SKILL",
                "skill": fallback_name,
                "input": fallback_arg[:200],
                "evidence_ids": [source["id"] for source in added_sources],
                "observation_preview": observation[:200],
                "skill_instructions_loaded": loaded_now,
            })
            messages.append({"role": "user", "content": observation_content})

        if state["requires_evidence"] and not evidence.has_evidence:
            final_answer = "抱歉，当前教材和图谱中没有找到足够证据来可靠回答这个问题。"
        else:
            final_answer = _generate_final_answer(
                client,
                settings,
                messages,
                state["task_plan"],
            )
            trace.append({
                "turn": turn + 1,
                "action": "FORCE_FINISH",
                "reason": "max_iter exhausted",
            })
        return {
            "observations": observations,
            "messages": messages,
            "trace": trace,
            "disclosed_skills": disclosed,
            "final_answer": final_answer,
            "route": "verify",
        }

    def verify(state: TeachingState) -> dict:
        evidence = _evidence_from_observations(state.get("observations", []))
        if not state["requires_evidence"] or not evidence.has_evidence:
            return {"route": "finalize"}
        verification = _verify_answer(
            client,
            settings,
            state["question"],
            state["task_plan"],
            state["final_answer"],
            evidence,
            state.get("observations", []),
            allowed_skills=state["available_skills"],
        )
        trace = list(state["trace"])
        trace.append({
            "turn": state.get("turn", 0) + 2,
            "action": "VERIFY",
            "reason_summary": "检查任务要求覆盖度、代码完整性和证据支持",
            "status": verification["status"],
            "issues": verification["issues"],
            "missing_requirements": verification["missing_requirements"],
            "fallback_reason": verification.get("fallback_reason", ""),
            "output_preview": verification.get("output_preview", ""),
        })
        route = {
            "pass": "finalize",
            "revise": "revise",
            "retrieve_more": "verify_retrieval",
        }[verification["status"]]
        return {"verification": verification, "trace": trace, "route": route}

    def revise(state: TeachingState) -> dict:
        verification = state["verification"]
        evidence = _evidence_from_observations(state.get("observations", []))
        trace = list(state["trace"])
        revised = _clean_answer(verification.get("revised_answer", ""))
        if revised:
            final_answer = revised
            reason = "根据验证结果修正回答，不新增证据外事实"
        else:
            final_answer = _generate_final_answer(
                client,
                settings,
                state["messages"],
                state["task_plan"],
                verification_feedback=verification["issues"],
            )
            reason = "验证要求修订但未返回正文，重新生成一次"
        remaining_issues = _deterministic_answer_issues(
            state["task_plan"], final_answer, evidence
        )
        trace.append({
            "turn": state.get("turn", 0) + 3,
            "action": "REVISE",
            "reason_summary": reason,
            "remaining_issues": remaining_issues,
        })
        if remaining_issues:
            final_answer += (
                "\n\n> 证据说明：修订后仍未完整覆盖："
                + "；".join(remaining_issues)
                + "。以上回答仅保留现有证据能够支持的部分。"
            )
        return {"final_answer": final_answer, "trace": trace, "route": "finalize"}

    def verify_retrieval(state: TeachingState) -> dict:
        verification = state["verification"]
        verify_skill = verification.get("suggested_skill") or _fallback_skill(
            state["available_skills"]
        )
        verify_query = verification.get("suggested_query") or (
            state["question"] + " " + " ".join(verification["missing_requirements"])
        )
        verify_key = _call_key(verify_skill, verify_query)
        seen_calls = list(state.get("seen_calls", []))
        observations = list(state.get("observations", []))
        messages = list(state["messages"])
        trace = list(state["trace"])
        disclosed = list(state.get("disclosed_skills", []))
        evidence = _evidence_from_observations(observations)
        added_sources = []
        loaded_now = False

        if verify_skill and verify_key not in seen_calls:
            seen_calls.append(verify_key)
            observation = _execute_skill(verify_skill, verify_query, skills)
            observations.append({
                "skill": verify_skill,
                "input": verify_query,
                "text": observation,
            })
            added_sources = evidence.add_observation(verify_skill, observation)
            observation_content, loaded_now, disclosed = observation_message(
                verify_skill,
                observation,
                disclosed,
            )
            messages.append({"role": "user", "content": observation_content})
        trace.append({
            "turn": state.get("turn", 0) + 3,
            "action": "VERIFY_RETRIEVAL",
            "skill": verify_skill,
            "input": verify_query[:200],
            "evidence_ids": [source["id"] for source in added_sources],
            "result": "executed" if added_sources else "no_new_evidence_or_duplicate",
            "skill_instructions_loaded": loaded_now,
        })
        return {
            "seen_calls": seen_calls,
            "observations": observations,
            "messages": messages,
            "trace": trace,
            "disclosed_skills": disclosed,
            "route": "regenerate",
        }

    def regenerate(state: TeachingState) -> dict:
        verification = state["verification"]
        evidence = _evidence_from_observations(state.get("observations", []))
        final_answer = _generate_final_answer(
            client,
            settings,
            state["messages"],
            state["task_plan"],
            verification_feedback=(
                verification["issues"] + verification["missing_requirements"]
            ),
        )
        remaining_issues = _deterministic_answer_issues(
            state["task_plan"], final_answer, evidence
        )
        trace = list(state["trace"])
        trace.append({
            "turn": state.get("turn", 0) + 4,
            "action": "REVISE",
            "reason_summary": "结合补充证据和验证反馈重新生成回答",
            "remaining_issues": remaining_issues,
        })
        if remaining_issues:
            final_answer += (
                "\n\n> 证据说明：当前材料仍未完整覆盖："
                + "；".join(remaining_issues)
                + "。以上回答仅保留现有证据能够支持的部分。"
            )
        return {"final_answer": final_answer, "trace": trace, "route": "finalize"}

    def finalize(state: TeachingState) -> dict:
        evidence = _evidence_from_observations(state.get("observations", []))
        selected_sources = evidence.select_for_answer(state["final_answer"])
        final_answer = append_references(state["final_answer"], selected_sources)
        concepts = _extract_concepts(state["trace"], state["question"])
        return {
            "final_answer": final_answer,
            "sources": selected_sources,
            "concepts_involved": concepts,
            "iterations": state.get("turn", 0),
        }

    graph = StateGraph(TeachingState)
    graph.add_node("prepare", prepare)
    graph.add_node("decide", decide)
    graph.add_node("execute_skill", execute_skill_node)
    graph.add_node("draft", draft)
    graph.add_node("fallback", fallback)
    graph.add_node("verify", verify)
    graph.add_node("revise", revise)
    graph.add_node("verify_retrieval", verify_retrieval)
    graph.add_node("regenerate", regenerate)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "decide")
    graph.add_conditional_edges(
        "decide",
        lambda state: state["route"],
        {
            "decide": "decide",
            "execute_skill": "execute_skill",
            "draft": "draft",
            "fallback": "fallback",
        },
    )
    graph.add_conditional_edges(
        "execute_skill",
        lambda state: state["route"],
        {"decide": "decide", "fallback": "fallback"},
    )
    graph.add_edge("draft", "verify")
    graph.add_edge("fallback", "verify")
    graph.add_conditional_edges(
        "verify",
        lambda state: state["route"],
        {
            "finalize": "finalize",
            "revise": "revise",
            "verify_retrieval": "verify_retrieval",
        },
    )
    graph.add_edge("revise", "finalize")
    graph.add_edge("verify_retrieval", "regenerate")
    graph.add_edge("regenerate", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def run_teaching_agent(
    question: str,
    max_iter: int = 3,
    session_history: list[dict] = None,
    prefs: dict = None,
) -> dict:
    """
    运行 LangGraph 教学 Agent，并保持原有 API 返回结构。

    Args:
        question: 学生问题
        max_iter: 最大轮数
        session_history: 最近 N 轮对话 [{"role": "user/assistant", "content": "..."}]

    Returns:
        {"answer": str, "trace": list, "iterations": int, "concepts_involved": list}
    """
    skills = load_all_skills()
    client = create_llm_client()
    settings = get_settings()
    graph = build_teaching_graph(skills, client, settings)
    state = graph.invoke({
        "question": question,
        "max_iter": max_iter,
        "session_history": session_history or [],
        "prefs": prefs or {},
    })
    return {
        "answer": state["final_answer"],
        "trace": state["trace"],
        "iterations": state["iterations"],
        "force_stopped": state["force_stopped"],
        "force_reason": state["force_reason"],
        "concepts_involved": state["concepts_involved"],
        "sources": state["sources"],
    }


def _clean_answer(raw: str) -> str:
    """去掉 LLM 输出中的 Thought/Action 格式残留，只保留答案正文。"""
    _, action_name, action_arg = _parse_action(raw)
    if action_name.upper() in ("FINISH", "FINAL_ANSWER") and action_arg:
        return action_arg

    # 兼容模型把未转义的多行正文放进 answer 字段所形成的非法 JSON。
    candidate = raw.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)
    answer_start = re.search(r'"answer"\s*:\s*"', candidate)
    if answer_start:
        wrapped_answer = candidate[answer_start.end():]
        closing = re.search(r'"\s*}\s*$', wrapped_answer, flags=re.DOTALL)
        if closing:
            wrapped_answer = wrapped_answer[:closing.start()]
        wrapped_answer = (
            wrapped_answer
            .replace("\\n", "\n")
            .replace("\\\"", '"')
            .replace("\\\\", "\\")
            .strip()
        )
        if wrapped_answer:
            return wrapped_answer

    lines = raw.split("\n")
    cleaned = []
    skip_suffix = False
    for line in lines:
        stripped = line.strip()
        # 跳过 Thought: / Action: / FINISH[ 开头的行
        if stripped.startswith("Thought") and ":" in stripped[:15]:
            continue
        if stripped.startswith("Action") and ":" in stripped[:15]:
            # Action: FINISH[xxx] → 提取方括号内容
            m = re.match(r"Action:\s*FINISH\s*\[(.*)", stripped, re.DOTALL)
            if m:
                cleaned.append(m.group(1))
                skip_suffix = False
            continue
        if stripped.startswith("Observation") and ":" in stripped[:15]:
            continue
        if not skip_suffix:
            cleaned.append(line)
    # 如果清理后为空（全是 ReAct 格式），尝试提取最后一段非格式文本
    result = "\n".join(cleaned).strip()
    if result.endswith("]"):
        result = result[:-1]
    if len(result) < 20:
        # 兜底：去掉首行 Thought 和末行 Action 后返回
        fallback_lines = [l for l in raw.split("\n")
                         if not l.strip().startswith("Thought")
                         and not l.strip().startswith("Action")
                         and not l.strip().startswith("Observation")]
        result = "\n".join(fallback_lines).strip()
    return result if len(result) > 20 else raw


def _extract_concepts(trace: list[dict], question: str) -> list[str]:
    """基于图谱实体词典提取概念，避免把完整检索句当作节点名。"""
    entities_path = Path(__file__).resolve().parent.parent / "kb" / "graph" / "entities.json"
    try:
        entities = json.loads(entities_path.read_text(encoding="utf-8"))
        known_names = [str(item.get("name", "")) for item in entities if item.get("name")]
    except (OSError, json.JSONDecodeError):
        known_names = []

    search_texts = [question]
    for t in trace:
        if t.get("input"):
            search_texts.append(str(t["input"]))

    normalized_sources = [re.sub(r"[·\s]", "", text) for text in search_texts]
    matches = []
    for name in sorted(known_names, key=len, reverse=True):
        normalized_name = re.sub(r"[·\s]", "", name)
        if normalized_name and any(normalized_name in source for source in normalized_sources):
            # 已匹配具体操作时不再重复加入同名的短结构词。
            if not any(name in existing or existing in name for existing in matches):
                matches.append(name)

    return matches[:5]
