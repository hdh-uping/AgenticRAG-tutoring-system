from types import SimpleNamespace

import app.preferences as preferences


def _fake_client(content: str):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )
    completions = SimpleNamespace(create=lambda **kwargs: response)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def test_llm_preferences_are_parsed_and_whitelisted(monkeypatch):
    monkeypatch.setattr(
        preferences,
        "create_llm_client",
        lambda: _fake_client(
            '```json\n{"preferences":{"depth":"beginner","show_code":"idea",'
            '"style":"academic","response_length":"detailed","unknown":"x"}}\n```'
        ),
    )

    assert preferences.extract_preferences("我刚入门，请正式、详细地讲思路") == {
        "depth": "beginner",
        "show_code": "idea",
        "style": "academic",
        "response_length": "detailed",
    }


def test_plain_knowledge_question_skips_llm(monkeypatch):
    def should_not_be_called():
        raise AssertionError("普通知识问题不应调用偏好提取 LLM")

    monkeypatch.setattr(preferences, "create_llm_client", should_not_be_called)
    assert preferences.extract_preferences("栈和队列有什么区别？") == {}
    assert preferences.extract_preferences("我希望了解栈是什么") == {}


def test_semantic_preference_signals_enter_llm_path(monkeypatch):
    monkeypatch.setattr(
        preferences,
        "create_llm_client",
        lambda: _fake_client('{"preferences":{"depth":"beginner"}}'),
    )
    assert preferences.extract_preferences("我基础比较薄弱，希望以后多铺垫") == {
        "depth": "beginner"
    }


def test_current_turn_modifier_is_left_for_llm_to_reject(monkeypatch):
    monkeypatch.setattr(
        preferences,
        "create_llm_client",
        lambda: _fake_client('{"preferences":{}}'),
    )
    assert preferences.should_extract_preferences("这次回答简短一点") is True
    assert preferences.extract_preferences("这次回答简短一点") == {}


def test_invalid_llm_values_are_ignored(monkeypatch):
    monkeypatch.setattr(
        preferences,
        "create_llm_client",
        lambda: _fake_client(
            '{"preferences":{"depth":"expert","style":"friendly"}}'
        ),
    )
    assert preferences.extract_preferences("讲专业些") == {}


def test_llm_failure_does_not_block_chat_flow(monkeypatch):
    def fail():
        raise TimeoutError("timeout")

    monkeypatch.setattr(preferences, "create_llm_client", fail)
    assert preferences.extract_preferences("以后回答简短一些") == {}
