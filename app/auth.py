"""本地用户名/密码认证与不透明 Bearer Token 管理。"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import get_settings
from app.db import (
    create_user,
    get_auth_token_user,
    get_user_credentials,
    revoke_auth_token,
    save_auth_token,
)


PASSWORD_ITERATIONS = 600_000
_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class AuthenticatedUser:
    user_id: str
    token_hash: str


def _password_digest(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS
    )


def _token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def register(user_id: str, password: str) -> dict | None:
    salt = secrets.token_bytes(16)
    if not create_user(user_id, salt, _password_digest(password, salt)):
        return None
    return issue_token(user_id)


def login(user_id: str, password: str) -> dict | None:
    credentials = get_user_credentials(user_id)
    if credentials is None:
        # 对不存在的用户也执行一次相同量级计算，减少用户名枚举侧信道。
        salt = bytes(16)
        _password_digest(password, salt)
        return None
    salt, expected = credentials
    if not hmac.compare_digest(_password_digest(password, salt), expected):
        return None
    return issue_token(user_id)


def issue_token(user_id: str) -> dict:
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(
        hours=get_settings().auth_token_ttl_hours
    )
    save_auth_token(
        _token_digest(token),
        user_id,
        expires_at.strftime("%Y-%m-%d %H:%M:%S"),
    )
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_at": expires_at.isoformat(),
        "user_id": user_id,
    }


def require_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> AuthenticatedUser:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少登录令牌",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token_hash = _token_digest(credentials.credentials)
    user_id = get_auth_token_user(token_hash)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="登录令牌无效或已过期",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return AuthenticatedUser(user_id=user_id, token_hash=token_hash)


def logout(user: AuthenticatedUser) -> None:
    revoke_auth_token(user.token_hash)
