from types import SimpleNamespace

import app.recommendation_agent as recommendation_agent


def test_recommendation_agent_has_its_own_langgraph():
    graph = recommendation_agent.build_recommendation_graph(
        [], SimpleNamespace(), SimpleNamespace()
    )
    nodes = set(graph.get_graph().nodes)
    assert {"prepare", "decide", "related_concepts", "fallback", "finalize"} <= nodes
    assert "execute_skill" not in nodes


def test_independent_agent_calls_related_skill_then_selects_candidates(monkeypatch):
    responses = iter([
        '{"reason_summary":"先查询图谱候选","action":"related_concepts","input":"顺序表·插入"}',
        '{"reason_summary":"删除和取元素适合作为后续","action":"finish",'
        '"recommendations":['
        '{"concept":"顺序表·删除","reason":"与插入形成对照，可以继续理解元素移动"},'
        '{"concept":"顺序表·取元素","reason":"承接顺序表按下标访问的特点"}]}'
    ])

    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=next(responses))
            )])

    monkeypatch.setattr(
        recommendation_agent,
        "create_llm_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions())),
    )
    monkeypatch.setattr(recommendation_agent, "load_all_skills", lambda: [])
    monkeypatch.setattr(
        recommendation_agent,
        "execute_skill",
        lambda name, arg, skills: (
            "你可能还想了解：顺序表·初始化、顺序表·删除、顺序表·取元素。"
        ),
    )

    result = recommendation_agent.run_recommendation_agent(
        ["顺序表·插入"],
        question="如何插入？",
        answer="已经讲解插入操作。",
    )

    assert "## 推荐继续学习" in result["advice"]
    assert "**顺序表·删除**：与插入形成对照" in result["advice"]
    assert "**顺序表·取元素**：承接顺序表按下标访问" in result["advice"]
    assert [item["action"] for item in result["trace"]] == ["related_concepts", "finish"]
    assert all(item["agent"] == "recommendation" for item in result["trace"])


def test_recommendation_agent_rejects_hallucinated_candidate(monkeypatch):
    responses = iter([
        '{"action":"related_concepts","input":"顺序表"}',
        '{"action":"finish","recommendations":['
        '{"concept":"B树","reason":"继续学习树结构"}]}',
    ])

    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=next(responses))
            )])

    monkeypatch.setattr(
        recommendation_agent,
        "create_llm_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions())),
    )
    monkeypatch.setattr(recommendation_agent, "load_all_skills", lambda: [])
    monkeypatch.setattr(
        recommendation_agent,
        "execute_skill",
        lambda name, arg, skills: "你可能还想了解：顺序表·插入、顺序表·删除。",
    )

    result = recommendation_agent.run_recommendation_agent(["顺序表"])
    assert result["advice"] == ""


def test_format_failure_fallback_balances_comparison_concepts(monkeypatch):
    responses = iter([
        '{"action":"related_concepts","input":"队列, 栈"}',
        '{"reason_summary":"被截断',
    ])

    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=next(responses))
            )])

    monkeypatch.setattr(
        recommendation_agent,
        "create_llm_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions())),
    )
    monkeypatch.setattr(recommendation_agent, "load_all_skills", lambda: [])
    monkeypatch.setattr(
        recommendation_agent,
        "execute_skill",
        lambda name, arg, skills: (
            "你可能还想了解：队列·入队、后缀表达式求值、队列·出队、"
            "栈·入栈、队列·初始化、栈·出栈。"
        ),
    )

    result = recommendation_agent.run_recommendation_agent(
        ["队列", "栈"], question="栈和队列有什么区别？"
    )
    assert "**队列·入队**" in result["advice"]
    assert "**栈·入栈**" in result["advice"]
    assert "具体步骤" in result["advice"]
