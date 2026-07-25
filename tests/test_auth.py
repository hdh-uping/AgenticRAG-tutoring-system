from pathlib import Path

from fastapi.testclient import TestClient

import app.db as db
import app.main as main


def _use_temporary_database(monkeypatch, tmp_path):
    if db._sqlite_conn is not None:
        db._sqlite_conn.close()
    monkeypatch.setattr(db, "_sqlite_conn", None)
    monkeypatch.setattr(db, "DB_PATH", Path(tmp_path) / "sessions.db")


def test_register_login_identity_and_logout(monkeypatch, tmp_path):
    _use_temporary_database(monkeypatch, tmp_path)
    client = TestClient(main.app)
    credentials = {"user_id": "alice", "password": "correct-password"}

    registered = client.post("/auth/register", json=credentials)
    assert registered.status_code == 201
    token = registered.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    assert client.get("/auth/me", headers=headers).json() == {"user_id": "alice"}

    assert client.post("/auth/register", json=credentials).status_code == 409
    assert client.post("/auth/login", json={
        **credentials, "password": "wrong-password"
    }).status_code == 401
    assert client.post("/auth/login", json=credentials).status_code == 200

    assert client.post("/auth/logout", headers=headers).status_code == 204
    assert client.get("/auth/me", headers=headers).status_code == 401
