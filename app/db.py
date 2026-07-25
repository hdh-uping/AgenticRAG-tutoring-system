"""SQLite 持久化：会话、完整消息历史和跨会话用户偏好。"""
import json
import re
import sqlite3
import threading
import uuid
from pathlib import Path


DB_PATH = Path(__file__).resolve().parent.parent / "data" / "sessions.db"

_sqlite_conn = None
_db_lock = threading.RLock()


def _clean_title(value: str | None, fallback: str = "新会话") -> str:
    title = re.sub(r"\s+", " ", value or "").strip()
    return (title[:40] or fallback)


def _title_from_question(question: str) -> str:
    return _clean_title(question)


def _get_sqlite():
    global _sqlite_conn
    with _db_lock:
        if _sqlite_conn is None:
            DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            _sqlite_conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
            _sqlite_conn.execute("PRAGMA foreign_keys = ON")
            _sqlite_conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '新会话',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            _sqlite_conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            _sqlite_conn.execute("""
                CREATE TABLE IF NOT EXISTS user_prefs (
                    user_id TEXT PRIMARY KEY,
                    prefs_json TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            _sqlite_conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    password_salt BLOB NOT NULL,
                    password_hash BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            _sqlite_conn.execute("""
                CREATE TABLE IF NOT EXISTS auth_tokens (
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
                )
            """)
            _sqlite_conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session ON messages(session_id)"
            )
            _sqlite_conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sessions_user_updated "
                "ON sessions(user_id, updated_at DESC)"
            )
            _sqlite_conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_auth_tokens_user "
                "ON auth_tokens(user_id)"
            )
            _backfill_legacy_sessions(_sqlite_conn)
            _sqlite_conn.commit()
    return _sqlite_conn


def _backfill_legacy_sessions(db: sqlite3.Connection) -> None:
    """把旧版只有 messages、没有 sessions 的数据迁移为 default 用户会话。"""
    session_ids = db.execute(
        "SELECT DISTINCT session_id FROM messages "
        "WHERE session_id NOT IN (SELECT id FROM sessions)"
    ).fetchall()
    for (session_id,) in session_ids:
        first_user = db.execute(
            "SELECT content FROM messages "
            "WHERE session_id = ? AND role = 'user' ORDER BY id LIMIT 1",
            (session_id,),
        ).fetchone()
        bounds = db.execute(
            "SELECT MIN(created_at), MAX(created_at) FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        created_at, updated_at = bounds or (None, None)
        db.execute(
            "INSERT OR IGNORE INTO sessions "
            "(id, user_id, title, created_at, updated_at) "
            "VALUES (?, 'default', ?, COALESCE(?, CURRENT_TIMESTAMP), "
            "COALESCE(?, CURRENT_TIMESTAMP))",
            (
                session_id,
                _title_from_question(first_user[0]) if first_user else "历史会话",
                created_at,
                updated_at,
            ),
        )


def _ensure_session(
    db: sqlite3.Connection,
    session_id: str,
    user_id: str = "default",
    title: str | None = None,
) -> None:
    db.execute(
        "INSERT OR IGNORE INTO sessions (id, user_id, title) VALUES (?, ?, ?)",
        (session_id, user_id, _clean_title(title)),
    )


def create_session(
    user_id: str,
    title: str | None = None,
    session_id: str | None = None,
) -> dict:
    """创建并返回一个属于 user_id 的持久化会话。"""
    new_id = session_id or uuid.uuid4().hex
    with _db_lock:
        db = _get_sqlite()
        db.execute(
            "INSERT INTO sessions (id, user_id, title) VALUES (?, ?, ?)",
            (new_id, user_id, _clean_title(title)),
        )
        db.commit()
    return get_session(new_id)


def _session_row_to_dict(row) -> dict | None:
    if not row:
        return None
    return {
        "session_id": row[0],
        "user_id": row[1],
        "title": row[2],
        "created_at": row[3],
        "updated_at": row[4],
        "message_count": int(row[5]),
        "last_message": row[6] or "",
    }


_SESSION_SELECT = """
    SELECT s.id, s.user_id, s.title, s.created_at, s.updated_at,
           COUNT(m.id) AS message_count,
           COALESCE((
               SELECT content FROM messages latest
               WHERE latest.session_id = s.id
               ORDER BY latest.id DESC LIMIT 1
           ), '') AS last_message
    FROM sessions s
    LEFT JOIN messages m ON m.session_id = s.id
"""


def get_session(session_id: str) -> dict | None:
    with _db_lock:
        db = _get_sqlite()
        row = db.execute(
            _SESSION_SELECT + " WHERE s.id = ? GROUP BY s.id",
            (session_id,),
        ).fetchone()
    return _session_row_to_dict(row)


def list_sessions(user_id: str, limit: int = 50, offset: int = 0) -> list[dict]:
    """按最近更新时间倒序列出用户会话。"""
    with _db_lock:
        db = _get_sqlite()
        rows = db.execute(
            _SESSION_SELECT
            + " WHERE s.user_id = ? GROUP BY s.id "
              "ORDER BY s.updated_at DESC, s.id DESC LIMIT ? OFFSET ?",
            (user_id, limit, offset),
        ).fetchall()
    return [_session_row_to_dict(row) for row in rows]


def rename_session(session_id: str, title: str) -> dict | None:
    with _db_lock:
        db = _get_sqlite()
        db.execute(
            "UPDATE sessions SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (_clean_title(title), session_id),
        )
        db.commit()
    return get_session(session_id)


def delete_session(session_id: str) -> bool:
    """删除会话及全部消息。"""
    with _db_lock:
        db = _get_sqlite()
        exists = db.execute(
            "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not exists:
            return False
        db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        db.commit()
    return True


def get_session_messages(
    session_id: str,
    limit: int | None = None,
    offset: int = 0,
) -> list[dict]:
    """按时间正序读取会话消息；不传 limit 时返回完整历史。"""
    query = (
        "SELECT id, role, content, metadata, created_at "
        "FROM messages WHERE session_id = ? ORDER BY id"
    )
    params: list = [session_id]
    if limit is not None:
        query += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])
    with _db_lock:
        rows = _get_sqlite().execute(query, params).fetchall()
    messages = []
    for message_id, role, content, metadata, created_at in rows:
        try:
            parsed_metadata = json.loads(metadata or "{}")
        except json.JSONDecodeError:
            parsed_metadata = {}
        messages.append({
            "id": message_id,
            "role": role,
            "content": content,
            "metadata": parsed_metadata,
            "created_at": created_at,
        })
    return messages


def get_session_history(session_id: str, limit: int | None = 10) -> list[dict]:
    """获取用于模型上下文的历史，保持旧调用格式。"""
    if limit is None:
        messages = get_session_messages(session_id)
    else:
        with _db_lock:
            rows = _get_sqlite().execute(
                "SELECT role, content FROM messages WHERE session_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [{"role": row[0], "content": row[1]} for row in reversed(rows)]
    return [{"role": item["role"], "content": item["content"]} for item in messages]


def get_context_history(
    session_id: str,
    max_messages: int = 40,
    max_chars: int = 30000,
) -> list[dict]:
    """从完整历史中取最近上下文窗口，避免无限会话超过模型上下文。"""
    history = get_session_history(session_id, limit=max_messages)
    selected = []
    used = 0
    for message in reversed(history):
        content = message["content"]
        remaining = max_chars - used
        if remaining <= 0:
            break
        if len(content) > remaining:
            if selected:
                break
            content = content[-remaining:]
        selected.append({"role": message["role"], "content": content})
        used += len(content)
    return list(reversed(selected))


def save_message(
    session_id: str,
    role: str,
    content: str,
    metadata: dict | None = None,
    user_id: str = "default",
) -> None:
    """保存单条消息；保留给兼容调用和测试使用。"""
    with _db_lock:
        db = _get_sqlite()
        _ensure_session(db, session_id, user_id=user_id, title=content if role == "user" else None)
        db.execute(
            "INSERT INTO messages (session_id, role, content, metadata) VALUES (?, ?, ?, ?)",
            (session_id, role, content, json.dumps(metadata or {}, ensure_ascii=False)),
        )
        if role == "user":
            db.execute(
                "UPDATE sessions SET "
                "title = CASE WHEN title = '新会话' THEN ? ELSE title END, "
                "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (_title_from_question(content), session_id),
            )
        else:
            db.execute(
                "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (session_id,),
            )
        db.commit()


def save_exchange(
    session_id: str,
    user_id: str,
    question: str,
    answer: str,
    assistant_metadata: dict | None = None,
) -> None:
    """在一个事务中成对保存问题和回答，并更新会话标题/时间。"""
    with _db_lock:
        db = _get_sqlite()
        _ensure_session(db, session_id, user_id=user_id, title=question)
        db.execute(
            "INSERT INTO messages (session_id, role, content, metadata) "
            "VALUES (?, 'user', ?, '{}')",
            (session_id, question),
        )
        db.execute(
            "INSERT INTO messages (session_id, role, content, metadata) "
            "VALUES (?, 'assistant', ?, ?)",
            (session_id, answer, json.dumps(assistant_metadata or {}, ensure_ascii=False)),
        )
        db.execute(
            "UPDATE sessions SET "
            "title = CASE WHEN title = '新会话' THEN ? ELSE title END, "
            "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (_title_from_question(question), session_id),
        )
        db.commit()


DEFAULT_PREFS = {
    "depth": "intermediate",
    "show_code": "full",
    "style": "casual",
    "response_length": "balanced",
}
PREF_OPTIONS = {
    "depth": {"beginner", "intermediate", "advanced"},
    "show_code": {"full", "idea"},
    "style": {"casual", "academic"},
    "response_length": {"concise", "balanced", "detailed"},
}


def save_prefs(user_id: str, prefs: dict) -> None:
    """保存跨会话用户偏好，只更新合法字段。"""
    with _db_lock:
        db = _get_sqlite()
        current = get_prefs(user_id)
        current.update(prefs)
        cleaned = {
            key: current[key]
            for key in DEFAULT_PREFS
            if current.get(key) in PREF_OPTIONS[key]
        }
        for key, default in DEFAULT_PREFS.items():
            cleaned.setdefault(key, default)
        db.execute(
            "INSERT OR REPLACE INTO user_prefs "
            "(user_id, prefs_json, updated_at) VALUES (?, ?, datetime('now'))",
            (user_id, json.dumps(cleaned, ensure_ascii=False)),
        )
        db.commit()


def get_prefs(user_id: str) -> dict:
    """获取跨会话偏好；新用户返回默认值。"""
    with _db_lock:
        row = _get_sqlite().execute(
            "SELECT prefs_json FROM user_prefs WHERE user_id = ?", (user_id,)
        ).fetchone()
    if row:
        try:
            stored = json.loads(row[0])
        except json.JSONDecodeError:
            stored = {}
        return {
            key: stored.get(key, default)
            if stored.get(key, default) in PREF_OPTIONS[key] else default
            for key, default in DEFAULT_PREFS.items()
        }
    return dict(DEFAULT_PREFS)


def create_user(user_id: str, password_salt: bytes, password_hash: bytes) -> bool:
    """创建本地用户；用户名已存在时返回 False。"""
    with _db_lock:
        db = _get_sqlite()
        try:
            db.execute(
                "INSERT INTO users (user_id, password_salt, password_hash) "
                "VALUES (?, ?, ?)",
                (user_id, password_salt, password_hash),
            )
            db.commit()
            return True
        except sqlite3.IntegrityError:
            db.rollback()
            return False


def get_user_credentials(user_id: str) -> tuple[bytes, bytes] | None:
    with _db_lock:
        row = _get_sqlite().execute(
            "SELECT password_salt, password_hash FROM users WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    return (bytes(row[0]), bytes(row[1])) if row else None


def save_auth_token(token_hash: str, user_id: str, expires_at: str) -> None:
    with _db_lock:
        db = _get_sqlite()
        db.execute("DELETE FROM auth_tokens WHERE expires_at <= CURRENT_TIMESTAMP")
        db.execute(
            "INSERT INTO auth_tokens (token_hash, user_id, expires_at) "
            "VALUES (?, ?, ?)",
            (token_hash, user_id, expires_at),
        )
        db.commit()


def get_auth_token_user(token_hash: str) -> str | None:
    with _db_lock:
        row = _get_sqlite().execute(
            "SELECT user_id FROM auth_tokens "
            "WHERE token_hash = ? AND expires_at > CURRENT_TIMESTAMP",
            (token_hash,),
        ).fetchone()
    return str(row[0]) if row else None


def revoke_auth_token(token_hash: str) -> bool:
    with _db_lock:
        db = _get_sqlite()
        cursor = db.execute(
            "DELETE FROM auth_tokens WHERE token_hash = ?", (token_hash,)
        )
        db.commit()
    return cursor.rowcount > 0
