import app.workflow as workflow


def _teaching_result(*, concepts=None):
    return {
        "answer": "教学回答",
        "trace": [{"action": "PLAN"}],
        "iterations": 2,
        "concepts_involved": concepts or [],
        "sources": [{"id": "线性表_chunk_001"}],
    }


def test_parent_graph_orchestrates_two_independent_agent_subgraphs():
    calls = []

    def teaching_runner(**kwargs):
        calls.append(("teaching", kwargs))
        return _teaching_result(concepts=["顺序表"])

    def recommendation_runner(concepts, **kwargs):
        calls.append(("recommendation", {"concepts": concepts, **kwargs}))
        return {
            "advice": "## 推荐继续学习\n\n- **单链表**：用于对比。",
            "trace": [{"agent": "recommendation", "action": "finish"}],
            "iterations": 2,
        }

    graph = workflow.build_tutoring_workflow(
        teaching_runner=teaching_runner,
        recommendation_runner=recommendation_runner,
    )
    nodes = set(graph.get_graph().nodes)
    assert {
        "prepare_context",
        "teaching_agent",
        "recommendation_agent",
        "recommendation_fallback",
        "assemble_response",
    } <= nodes

    state = graph.invoke({
        "question": "顺序表是什么？",
        "session_history": [{"role": "user", "content": "历史问题"}],
        "prefs": {"depth": "beginner"},
        "teaching_max_iter": 3,
        "recommendation_max_iter": 2,
    })

    assert [name for name, _ in calls] == ["teaching", "recommendation"]
    assert calls[0][1]["session_history"][0]["content"] == "历史问题"
    assert calls[1][1]["concepts"] == ["顺序表"]
    assert calls[1][1]["answer"] == "教学回答"
    assert "教学回答" in state["answer"]
    assert "推荐继续学习" in state["answer"]
    assert state["recommendation_trace"][0]["agent"] == "recommendation"


def test_parent_graph_isolates_recommendation_agent_failure():
    def teaching_runner(**kwargs):
        return _teaching_result(concepts=["顺序表"])

    def recommendation_runner(*args, **kwargs):
        raise RuntimeError("recommendation unavailable")

    graph = workflow.build_tutoring_workflow(
        teaching_runner=teaching_runner,
        recommendation_runner=recommendation_runner,
    )
    state = graph.invoke({"question": "顺序表是什么？"})

    assert state["answer"] == "教学回答"
    assert state["recommendation"] == ""
    assert state["recommendation_trace"][0]["action"] == "PARENT_GRAPH_FALLBACK"


def test_parent_graph_skips_recommendation_when_no_concept_was_extracted():
    called = False

    def recommendation_runner(*args, **kwargs):
        nonlocal called
        called = True
        return {"advice": "不应出现", "trace": [], "iterations": 1}

    graph = workflow.build_tutoring_workflow(
        teaching_runner=lambda **kwargs: _teaching_result(),
        recommendation_runner=recommendation_runner,
    )
    state = graph.invoke({"question": "你好"})

    assert called is False
    assert state["answer"] == "教学回答"
    assert state["recommendation_trace"] == []
