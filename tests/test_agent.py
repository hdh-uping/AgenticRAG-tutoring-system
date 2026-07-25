from types import SimpleNamespace

import app.agent as agent


def _pass_verification(*args, **kwargs):
    return {
        "status": "pass",
        "issues": [],
        "missing_requirements": [],
        "suggested_skill": "",
        "suggested_query": "",
        "revised_answer": "",
    }


def test_teaching_agent_is_compiled_as_explicit_langgraph():
    graph = agent.build_teaching_graph([], SimpleNamespace(), SimpleNamespace())
    nodes = set(graph.get_graph().nodes)
    assert {
        "prepare", "decide", "execute_skill", "draft", "fallback",
        "verify", "revise", "verify_retrieval", "regenerate", "finalize",
    } <= nodes


def test_parse_json_skill_action():
    raw = '{"reason_summary":"需要教材证据","action":"hybrid_retrieval","input":"顺序表插入"}'
    assert agent._parse_action(raw) == ("需要教材证据", "hybrid_retrieval", "顺序表插入")


def test_parse_fenced_json_finish():
    raw = '```json\n{"action":"finish","answer":"第一行\\n第二行"}\n```'
    assert agent._parse_action(raw) == ("", "FINISH", "第一行\n第二行")


def test_parse_legacy_multiline_finish():
    raw = "Thought: 证据充分\nAction: FINISH[第一行\n第二行]"
    assert agent._parse_action(raw) == ("证据充分", "FINISH", "第一行\n第二行")


def test_relevance_scores_do_not_force_stop_before_iteration_limit():
    assert agent._should_force_stop("rerank_score=0.9 rerank_score=0.8", 1) == (False, "")
    assert agent._should_force_stop(
        "rerank_score=0.9 rerank_score=0.8 rerank_score=0.7", 1
    ) == (False, "")
    stopped, reason = agent._should_force_stop("", 3)
    assert stopped is True
    assert "3 轮" in reason


def test_clean_answer_recovers_unescaped_multiline_json():
    raw = '{"reason_summary":"证据充分","action":"finish","answer":"第一行\n第二行。"}'
    assert agent._clean_answer(raw) == "第一行\n第二行。"


def test_task_plan_makes_code_completion_requirements_explicit():
    plan = agent._build_task_plan("稀疏矩阵压缩的代码如何实现")
    assert plan["question_type"] == "implementation"
    assert "required_skills" not in plan
    assert all("skill" not in step for step in plan["steps"])
    assert "function_code" in plan["checks"]
    assert any("完整代码" in requirement for requirement in plan["requirements"])


def test_task_plan_respects_explicit_no_code_request():
    plan = agent._build_task_plan("只讲稀疏矩阵压缩的实现思路，不需要代码")
    assert plan["question_type"] == "knowledge"
    assert "function_code" not in plan["checks"]
    assert not any("完整代码" in requirement for requirement in plan["requirements"])


def test_extract_concepts_uses_graph_vocabulary():
    trace = [{"action": "hybrid_retrieval", "input": "请解释顺序表插入的复杂度"}]
    concepts = agent._extract_concepts(trace, "顺序表怎么插入？")
    assert "顺序表·插入" in concepts
    assert "请解释顺序表插入的复杂度" not in concepts


def test_long_unstructured_answer_is_recorded_as_format_error(monkeypatch):
    answer = "这是一个足够长的直接回答。" * 30

    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=answer))]
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr(agent, "create_llm_client", lambda: fake_client)
    monkeypatch.setattr(agent, "load_all_skills", lambda: [])
    monkeypatch.setattr(agent, "build_skill_prompt", lambda skills: "")

    result = agent.run_teaching_agent("什么是线性表？", max_iter=1)
    assert "没有找到足够证据" in result["answer"]
    assert result["trace"][0]["action"] == "PLAN"
    assert result["trace"][1]["action"] == "FORMAT_ERROR"
    assert result["trace"][2]["action"] == "FALLBACK_SKILL"


def test_structured_skill_call_records_evidence(monkeypatch):
    responses = iter([
        '{"reason_summary":"需要教材证据","action":"hybrid_retrieval","input":"线性表定义"}',
        '{"reason_summary":"证据充分","action":"finish","answer":"线性表是有限序列。"}',
    ])

    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=next(responses))
            )])

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr(agent, "create_llm_client", lambda: fake_client)
    monkeypatch.setattr(agent, "load_all_skills", lambda: [{"name": "hybrid_retrieval"}])
    monkeypatch.setattr(agent, "build_skill_prompt", lambda skills: "")
    monkeypatch.setattr(agent, "_verify_answer", _pass_verification)
    monkeypatch.setattr(
        agent,
        "execute_skill",
        lambda name, arg, skills: (
            "[Chunk1 | id=线性表_chunk_003 | rerank_score=0.9000 | 第1页]\n教材证据"
        ),
    )

    result = agent.run_teaching_agent("什么是线性表？")
    assert result["answer"].startswith("线性表是有限序列。")
    assert "线性表_chunk_003" in result["answer"]
    retrieval = next(item for item in result["trace"] if item["action"] == "hybrid_retrieval")
    finish = next(item for item in result["trace"] if item["action"] == "FINISH")
    assert retrieval["evidence_ids"] == ["线性表_chunk_003"]
    assert retrieval["rerank_scores"] == [0.9]
    assert finish["result"] == "agent_decided_finish"


def test_knowledge_question_cannot_finish_before_retrieval(monkeypatch):
    responses = iter([
        '{"action":"finish","answer":"直接回答"}',
        '{"action":"graph_lookup","input":"顺序表"}',
        '{"action":"finish","answer":"有证据的回答"}',
    ])

    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=next(responses))
            )])

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr(agent, "create_llm_client", lambda: fake_client)
    monkeypatch.setattr(agent, "load_all_skills", lambda: [{"name": "graph_lookup"}])
    monkeypatch.setattr(agent, "build_skill_prompt", lambda skills: "")
    monkeypatch.setattr(agent, "_verify_answer", _pass_verification)
    monkeypatch.setattr(
        agent, "execute_skill", lambda name, arg, skills: "[数据结构] 顺序表\n  描述: 连续存储"
    )

    result = agent.run_teaching_agent("顺序表插入代码怎么写？")
    assert result["trace"][1]["result"] == "finish_rejected_no_evidence"
    assert result["trace"][2]["evidence_ids"] == ["顺序表"]
    assert result["answer"].startswith("有证据的回答")


def test_code_question_can_finish_with_valid_retrieval_evidence(monkeypatch):
    responses = iter([
        '{"action":"hybrid_retrieval","input":"顺序表插入完整代码"}',
        '{"action":"finish","answer":"教材中的完整实现。"}',
    ])

    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=next(responses))
            )])

    monkeypatch.setattr(
        agent,
        "create_llm_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions())),
    )
    monkeypatch.setattr(
        agent,
        "load_all_skills",
        lambda: [{"name": "graph_lookup"}, {"name": "hybrid_retrieval"}],
    )
    monkeypatch.setattr(agent, "build_skill_prompt", lambda skills: "")
    monkeypatch.setattr(agent, "_verify_answer", _pass_verification)
    monkeypatch.setattr(
        agent,
        "execute_skill",
        lambda name, arg, skills: (
            "[Chunk1 | id=线性表_chunk_019 | rerank_score=0.9000 | 第4页]\n完整代码"
        ),
    )

    result = agent.run_teaching_agent("请给出顺序表插入代码")
    assert result["answer"].startswith("教材中的完整实现。")
    assert not any(
        item.get("result") == "finish_rejected_missing_skill"
        for item in result["trace"]
    )


def test_follow_up_style_request_can_answer_without_evidence():
    history = [{"role": "assistant", "content": "较长的解释"}]
    assert agent._requires_evidence("请再简单一点", history) is False
    assert agent._requires_evidence("顺序表是什么", history) is True


def test_duplicate_skill_call_is_blocked(monkeypatch):
    responses = iter([
        '{"action":"graph_lookup","input":"顺序表"}',
        '{"action":"graph_lookup","input":"顺序表"}',
        '{"action":"finish","answer":"顺序表使用连续存储。"}',
    ])
    execution_count = 0

    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=next(responses))
            )])

    def fake_execute(name, arg, skills):
        nonlocal execution_count
        execution_count += 1
        return "[数据结构] 顺序表\n  描述: 连续存储"

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr(agent, "create_llm_client", lambda: fake_client)
    monkeypatch.setattr(agent, "load_all_skills", lambda: [{"name": "graph_lookup"}])
    monkeypatch.setattr(agent, "build_skill_prompt", lambda skills: "")
    monkeypatch.setattr(agent, "execute_skill", fake_execute)
    monkeypatch.setattr(agent, "_verify_answer", _pass_verification)

    result = agent.run_teaching_agent("顺序表插入代码怎么写？")
    assert execution_count == 1
    assert result["trace"][2]["result"] == "duplicate_call_blocked"


def test_deterministic_verifier_rejects_struct_definition_without_function():
    plan = agent._build_task_plan("稀疏矩阵压缩的代码如何实现")
    answer = """```c
typedef struct { int i, j, e; } Triple;
typedef struct { Triple data[100]; int mu, nu, tu; } TSMatrix;
```"""
    issues = agent._deterministic_answer_issues(plan, answer, agent.EvidencePool())
    assert any("完整实现" in issue for issue in issues)


def test_verification_can_retrieve_once_and_regenerate_complete_code(monkeypatch):
    responses = iter([
        '{"action":"graph_lookup","input":"三元组顺序表"}',
        '{"action":"finish","answer":"```c\\ntypedef struct { int i,j,e; } Triple;\\n```"}',
        """下面给出完整实现：
```c
int Compress(int rows, int cols, int a[rows][cols], TSMatrix *out) {
    if (!out) return -1;
    out->mu = rows; out->nu = cols; out->tu = 0;
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            if (a[i][j] != 0) out->data[++out->tu] = (Triple){i, j, a[i][j]};
    return 1;
}
```""",
    ])

    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=next(responses))
            )])

    def fake_execute(name, arg, skills):
        if name == "graph_lookup":
            return "[数据结构] 三元组顺序表\n  描述: 按行存储非零元素"
        return (
            "[Chunk1 | id=数组_chunk_022 | rerank_score=0.9900 | 第4页]\n"
            "三元组顺序表结构定义"
        )

    def needs_more(*args, **kwargs):
        return {
            "status": "retrieve_more",
            "issues": ["当前只有结构定义，没有压缩函数"],
            "missing_requirements": ["完整压缩函数"],
            "suggested_skill": "hybrid_retrieval",
            "suggested_query": "稀疏矩阵生成三元组表代码",
            "revised_answer": "",
        }

    monkeypatch.setattr(
        agent,
        "create_llm_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions())),
    )
    monkeypatch.setattr(
        agent,
        "load_all_skills",
        lambda: [{"name": "graph_lookup"}, {"name": "hybrid_retrieval"}],
    )
    monkeypatch.setattr(agent, "build_skill_prompt", lambda skills: "")
    monkeypatch.setattr(agent, "execute_skill", fake_execute)
    monkeypatch.setattr(agent, "_verify_answer", needs_more)

    result = agent.run_teaching_agent("稀疏矩阵压缩的代码如何实现")
    actions = [item["action"] for item in result["trace"]]
    assert actions[0] == "PLAN"
    assert "VERIFY" in actions
    assert "VERIFY_RETRIEVAL" in actions
    assert "REVISE" in actions
    assert "int Compress" in result["answer"]
    assert "证据说明" not in result["answer"]


def test_persisted_preferences_are_injected_into_system_prompt(monkeypatch):
    captured_messages = []

    class FakeCompletions:
        def create(self, **kwargs):
            captured_messages.extend(kwargs["messages"])
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=(
                    '{"action":"finish","answer":"你好，有什么可以帮助你的？"}'
                ))
            )])

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr(agent, "create_llm_client", lambda: fake_client)
    monkeypatch.setattr(agent, "load_all_skills", lambda: [])
    monkeypatch.setattr(agent, "build_skill_prompt", lambda skills: "")

    agent.run_teaching_agent("你好", prefs={
        "depth": "beginner",
        "show_code": "idea",
        "style": "academic",
        "response_length": "concise",
    })

    system_prompt = captured_messages[0]["content"]
    assert "学生是初学者" in system_prompt
    assert "只讲思路" in system_prompt
    assert "语气正式" in system_prompt
    assert "回答简洁" in system_prompt


def test_agent_uses_the_database_trimmed_history_without_second_truncation(monkeypatch):
    captured = []

    class FakeCompletions:
        def create(self, **kwargs):
            captured.extend(kwargs["messages"])
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=(
                    '{"action":"finish","answer":"你好，有什么可以帮助你的？"}'
                ))
            )])

    monkeypatch.setattr(
        agent,
        "create_llm_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions())),
    )
    monkeypatch.setattr(agent, "load_all_skills", lambda: [])
    monkeypatch.setattr(agent, "build_skill_prompt", lambda skills: "")
    history = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"历史{index}"}
        for index in range(10)
    ]

    agent.run_teaching_agent("你好", session_history=history)

    history_contents = [item["content"] for item in captured[1:-1]]
    assert history_contents == [f"历史{index}" for index in range(10)]
