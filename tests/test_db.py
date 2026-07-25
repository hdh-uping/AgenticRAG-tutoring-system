import sqlite3

import app.db as db


def _use_temporary_database(monkeypatch, tmp_path):
    if db._sqlite_conn is not None:
        db._sqlite_conn.close()
    monkeypatch.setattr(db, "_sqlite_conn", None)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "sessions.db")


def test_message_history_is_chronological(monkeypatch, tmp_path):
    _use_temporary_database(monkeypatch, tmp_path)
    db.save_message("session-1", "user", "问题")
    db.save_message("session-1", "assistant", "回答")
    assert db.get_session_history("session-1") == [
        {"role": "user", "content": "问题"},
        {"role": "assistant", "content": "回答"},
    ]


def test_invalid_preferences_fall_back_to_defaults(monkeypatch, tmp_path):
    _use_temporary_database(monkeypatch, tmp_path)
    db.save_prefs("student", {"depth": "invalid", "show_code": "idea"})
    assert db.get_prefs("student") == {
        "depth": "intermediate",
        "show_code": "idea",
        "style": "casual",
        "response_length": "balanced",
    }


def test_full_session_history_survives_connection_restart(monkeypatch, tmp_path):
    _use_temporary_database(monkeypatch, tmp_path)
    session = db.create_session("student", title="测试会话")
    session_id = session["session_id"]
    for index in range(3):
        db.save_exchange(session_id, "student", f"问题{index}", f"回答{index}")

    assert len(db.get_session_messages(session_id)) == 6
    assert db.get_session(session_id)["message_count"] == 6

    db._sqlite_conn.close()
    db._sqlite_conn = None

    assert [item["content"] for item in db.get_session_messages(session_id)] == [
        "问题0", "回答0", "问题1", "回答1", "问题2", "回答2",
    ]
    assert [item["content"] for item in db.get_context_history(
        session_id, max_messages=2, max_chars=100
    )] == ["问题2", "回答2"]


def test_preferences_are_persisted(monkeypatch, tmp_path):
    _use_temporary_database(monkeypatch, tmp_path)
    preferences = {
        "depth": "beginner",
        "show_code": "idea",
        "style": "academic",
        "response_length": "detailed",
    }
    db.save_prefs("student", preferences)
    assert db.get_prefs("student") == preferences


def test_session_listing_rename_and_delete(monkeypatch, tmp_path):
    _use_temporary_database(monkeypatch, tmp_path)
    session = db.create_session("student")
    session_id = session["session_id"]
    db.save_exchange(session_id, "student", "什么是线性表？", "线性表是有限序列。")

    listed = db.list_sessions("student")
    assert listed[0]["session_id"] == session_id
    assert listed[0]["message_count"] == 2
    assert db.rename_session(session_id, "线性表复习")["title"] == "线性表复习"
    assert db.delete_session(session_id) is True
    assert db.get_session(session_id) is None
    assert db.get_session_messages(session_id) == []


def test_legacy_messages_are_backfilled_as_default_user_sessions(monkeypatch, tmp_path):
    legacy_path = tmp_path / "sessions.db"
    connection = sqlite3.connect(legacy_path)
    connection.execute("""
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    connection.execute(
        "INSERT INTO messages (session_id, role, content) VALUES ('legacy', 'user', '旧问题')"
    )
    connection.commit()
    connection.close()

    _use_temporary_database(monkeypatch, tmp_path)

    session = db.get_session("legacy")
    assert session["user_id"] == "default"
    assert session["title"] == "旧问题"
    assert session["message_count"] == 1


def teardown_module():
    if db._sqlite_conn is not None:
        db._sqlite_conn.close()
        db._sqlite_conn = None
