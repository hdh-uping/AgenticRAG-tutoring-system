from fastapi.testclient import TestClient

import app.db as db
import app.main as main


def _use_temporary_database(monkeypatch, tmp_path):
    if db._sqlite_conn is not None:
        db._sqlite_conn.close()
    monkeypatch.setattr(db, "_sqlite_conn", None)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "sessions.db")


def _register(client, user_id: str) -> dict:
    response = client.post("/auth/register", json={
        "user_id": user_id,
        "password": "correct-password",
    })
    assert response.status_code == 201
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def test_session_can_be_reopened_and_continued_with_cross_session_prefs(
    monkeypatch, tmp_path
):
    _use_temporary_database(monkeypatch, tmp_path)
    calls = []

    def fake_workflow(question, session_history, prefs):
        calls.append({
            "question": question,
            "session_history": session_history,
            "prefs": prefs,
        })
        return {
            "answer": f"回答：{question}",
            "recommendation": "",
            "trace": [],
            "recommendation_trace": [],
            "iterations": 1,
            "concepts_involved": [],
            "sources": [],
        }

    monkeypatch.setattr(main, "run_tutoring_workflow", fake_workflow)
    monkeypatch.setattr(
        main,
        "extract_preferences",
        lambda question, current: {
            "depth": "beginner",
            "show_code": "idea",
            "response_length": "concise",
        } if "我是初学者" in question else {},
    )
    client = TestClient(main.app)
    headers = _register(client, "alice")

    first = client.post("/chat", headers=headers, json={
        "question": "我是初学者，请不要代码，回答简短一点：什么是线性表？",
    })
    assert first.status_code == 200
    session_id = first.json()["session_id"]
    assert calls[0]["session_history"] == []
    assert calls[0]["prefs"] == {
        "depth": "beginner",
        "show_code": "idea",
        "style": "casual",
        "response_length": "concise",
    }

    detail = client.get(f"/sessions/{session_id}", headers=headers)
    assert detail.status_code == 200
    assert [item["role"] for item in detail.json()["messages"]] == ["user", "assistant"]

    # 模拟服务重启后重新打开 SQLite，再使用同一 session_id 继续提问。
    db._sqlite_conn.close()
    db._sqlite_conn = None
    second = client.post("/chat", headers=headers, json={
        "session_id": session_id,
        "question": "请继续解释。",
    })
    assert second.status_code == 200
    assert len(calls[1]["session_history"]) == 2
    assert calls[1]["prefs"]["depth"] == "beginner"
    assert calls[1]["prefs"]["show_code"] == "idea"

    reopened = client.get(f"/sessions/{session_id}", headers=headers)
    assert reopened.json()["message_count"] == 4
    assert len(reopened.json()["messages"]) == 4

    sessions = client.get("/sessions", headers=headers)
    assert sessions.status_code == 200
    assert sessions.json()["sessions"][0]["session_id"] == session_id

    # 偏好属于用户而非会话，新开会话仍会注入。
    third = client.post("/chat", headers=headers, json={
        "question": "什么是栈？",
    })
    assert third.status_code == 200
    assert calls[2]["prefs"]["response_length"] == "concise"


def test_session_ownership_rename_delete_and_missing_session(monkeypatch, tmp_path):
    _use_temporary_database(monkeypatch, tmp_path)
    monkeypatch.setattr(
        main,
        "extract_preferences",
        lambda question, current: {"depth": "beginner"},
    )
    client = TestClient(main.app)
    alice_headers = _register(client, "alice")
    bob_headers = _register(client, "bob")
    created = client.post(
        "/sessions", headers=alice_headers, json={"title": "旧标题"}
    )
    assert created.status_code == 201
    session_id = created.json()["session_id"]

    forbidden = client.get(f"/sessions/{session_id}", headers=bob_headers)
    assert forbidden.status_code == 403

    renamed = client.patch(
        f"/sessions/{session_id}",
        headers=alice_headers,
        json={"title": "新标题"},
    )
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "新标题"

    missing = client.post("/chat", headers=alice_headers, json={
        "session_id": "not-found",
        "question": "以后从基础讲",
    })
    assert missing.status_code == 404
    assert db.get_prefs("alice")["depth"] == "intermediate"

    deleted = client.delete(f"/sessions/{session_id}", headers=alice_headers)
    assert deleted.status_code == 204
    assert client.get(
        f"/sessions/{session_id}", headers=alice_headers
    ).status_code == 404


def test_protected_endpoint_requires_bearer_token(monkeypatch, tmp_path):
    _use_temporary_database(monkeypatch, tmp_path)
    response = TestClient(main.app).get("/sessions")
    assert response.status_code == 401
