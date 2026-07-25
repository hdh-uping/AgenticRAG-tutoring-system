"""FastAPI 应用：调用双 Agent 总图并管理会话、偏好和数据层。"""
import logging
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Response, status
from pydantic import BaseModel, Field

from app.auth import AuthenticatedUser, login, logout, register, require_user
from app.config import ConfigurationError, get_settings
from app.readiness import check_readiness
from app.sources import get_source
from app.preferences import extract_preferences
from app.workflow import run_tutoring_workflow
from app.db import (
    create_session,
    delete_session,
    get_context_history,
    get_prefs,
    get_session,
    get_session_messages,
    list_sessions,
    rename_session,
    save_exchange,
    save_prefs,
)

app = FastAPI(title="Agentic RAG Tutoring System")
logger = logging.getLogger(__name__)


class AuthRequest(BaseModel):
    user_id: str = Field(
        min_length=3,
        max_length=64,
        pattern=r"^[A-Za-z0-9_.-]+$",
    )
    password: str = Field(min_length=8, max_length=128)


class AuthResponse(BaseModel):
    access_token: str
    token_type: str
    expires_at: str
    user_id: str


@app.post(
    "/auth/register",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
)
def register_user(req: AuthRequest):
    token = register(req.user_id, req.password)
    if token is None:
        raise HTTPException(status_code=409, detail="用户名已存在")
    return token


@app.post("/auth/login", response_model=AuthResponse)
def login_user(req: AuthRequest):
    token = login(req.user_id, req.password)
    if token is None:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    return token


@app.get("/auth/me")
def read_current_user(user: AuthenticatedUser = Depends(require_user)):
    return {"user_id": user.user_id}


@app.post("/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout_user(user: AuthenticatedUser = Depends(require_user)):
    logout(user)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    session_id: str = Field(default="", max_length=64)  # 空则自动创建


class ChatResponse(BaseModel):
    answer: str
    recommendation: str        # 关联知识点推荐
    session_id: str
    trace: list[dict]          # Agent 决策轨迹
    recommendation_trace: list[dict]
    iterations: int
    concepts_involved: list[str]
    sources: list[dict]


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, user: AuthenticatedUser = Depends(require_user)):
    # 1. 先验证已有会话归属，避免无效/越权请求产生偏好写入等副作用。
    if req.session_id:
        session = _require_owned_session(req.session_id, user.user_id)
    else:
        session = create_session(user.user_id, title=req.question)
    session_id = session["session_id"]

    # 2. 仅在本轮可能包含偏好时调用 LLM，提取结果对当前回答立即生效。
    current_prefs = get_prefs(user.user_id)
    inferred_prefs = extract_preferences(req.question, current_prefs)
    if inferred_prefs:
        save_prefs(user.user_id, inferred_prefs)
    prefs = get_prefs(user.user_id)

    # 3. 加载受预算限制的长期历史上下文。
    settings = get_settings()
    history = get_context_history(
        session_id,
        max_messages=settings.session_context_max_messages,
        max_chars=settings.session_context_max_chars,
    )

    # 4. 由一张父 LangGraph 编排教学 Agent 与推荐 Agent 两个独立子图。
    try:
        result = run_tutoring_workflow(
            question=req.question,
            session_history=history,
            prefs=prefs,
        )
    except ConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("双 Agent 工作流执行失败")
        raise HTTPException(status_code=503, detail="教学服务暂时不可用") from exc

    # 5. 父图已经完成结果组装；推荐失败时由图内降级边保留教学答案。
    final_answer = result["answer"]
    recommendation = result.get("recommendation", "")

    # 6. 工作流成功后在一个事务中成对保存，完整历史可跨进程、跨启动恢复。
    save_exchange(
        session_id,
        user.user_id,
        req.question,
        final_answer,
        {
            "iterations": result["iterations"],
            "concepts": result["concepts_involved"],
            "sources": result["sources"],
            "trace": result["trace"],
            "recommendation_trace": result.get("recommendation_trace", []),
            "inferred_prefs": inferred_prefs,
        },
    )

    return ChatResponse(
        answer=final_answer,
        recommendation=recommendation,
        session_id=session_id,
        trace=result["trace"],
        recommendation_trace=result.get("recommendation_trace", []),
        iterations=result["iterations"],
        concepts_involved=result["concepts_involved"],
        sources=result["sources"],
    )


@app.get("/sources/{chunk_id}")
def read_source(chunk_id: str):
    source = get_source(chunk_id)
    if source is None:
        raise HTTPException(status_code=404, detail="未找到该教材片段")
    return source


class PrefsRequest(BaseModel):
    depth: Literal["beginner", "intermediate", "advanced"] | None = None
    show_code: Literal["full", "idea"] | None = None
    style: Literal["casual", "academic"] | None = None
    response_length: Literal["concise", "balanced", "detailed"] | None = None


@app.get("/prefs")
def get_user_prefs(user: AuthenticatedUser = Depends(require_user)):
    return {"user_id": user.user_id, "prefs": get_prefs(user.user_id)}


@app.put("/prefs")
def set_user_prefs(
    req: PrefsRequest,
    user: AuthenticatedUser = Depends(require_user),
):
    updates = req.model_dump(exclude_none=True)
    save_prefs(user.user_id, updates)
    return {"user_id": user.user_id, "prefs": get_prefs(user.user_id)}


class SessionCreateRequest(BaseModel):
    title: str = Field(default="新会话", max_length=80)


class SessionRenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=80)


def _require_owned_session(session_id: str, user_id: str) -> dict:
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    if session["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="无权访问该会话")
    return session


@app.post("/sessions", status_code=status.HTTP_201_CREATED)
def new_session(
    req: SessionCreateRequest,
    user: AuthenticatedUser = Depends(require_user),
):
    """显式创建空会话；也可省略此步，首次 /chat 会自动创建。"""
    return create_session(user.user_id, title=req.title)


@app.get("/sessions")
def get_sessions(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    user: AuthenticatedUser = Depends(require_user),
):
    sessions = list_sessions(user.user_id, limit=limit, offset=offset)
    return {
        "user_id": user.user_id,
        "sessions": sessions,
        "limit": limit,
        "offset": offset,
    }


@app.get("/sessions/{session_id}")
def read_session(
    session_id: str,
    limit: int = Query(default=200, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    user: AuthenticatedUser = Depends(require_user),
):
    """打开会话并读取按时间正序排列的历史；长会话可用 offset 分页。"""
    session = _require_owned_session(session_id, user.user_id)
    messages = get_session_messages(session_id, limit=limit, offset=offset)
    return {
        **session,
        "messages": messages,
        "limit": limit,
        "offset": offset,
        "has_more": offset + len(messages) < session["message_count"],
    }


@app.patch("/sessions/{session_id}")
def update_session(
    session_id: str,
    req: SessionRenameRequest,
    user: AuthenticatedUser = Depends(require_user),
):
    _require_owned_session(session_id, user.user_id)
    return rename_session(session_id, req.title)


@app.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_session(
    session_id: str,
    user: AuthenticatedUser = Depends(require_user),
):
    _require_owned_session(session_id, user.user_id)
    delete_session(session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def readiness(response: Response):
    """检查关键依赖、模型文件、图谱和 Milvus collection，不调用外部 LLM。"""
    report = check_readiness()
    if not report["ready"]:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "ready" if report["ready"] else "not_ready",
        "components": report["components"],
        "details": report["details"],
    }
