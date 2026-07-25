"""应用配置。

所有凭据和机器相关路径都从环境变量读取。项目根目录下的 ``.env`` 会在
本地开发时自动加载；生产环境应由部署平台注入环境变量。
"""
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = PROJECT_DIR.parent / "models" / "Qwen"


class ConfigurationError(RuntimeError):
    """缺少运行当前能力所需的配置。"""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-v4-flash"
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2

    # 登录令牌有效期。令牌原文只返回给客户端，SQLite 仅保存 SHA-256 摘要。
    auth_token_ttl_hours: int = 24 * 30

    neo4j_uri: str = "neo4j://127.0.0.1:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    embed_model_path: Path = DEFAULT_MODELS_DIR / "Qwen3-Embedding-0.6B"
    rerank_model_path: Path = DEFAULT_MODELS_DIR / "Qwen3-Reranker-0.6B"
    vlm_model_path: Path = DEFAULT_MODELS_DIR / "Qwen3-VL-4B-Instruct"

    milvus_db_path: Path = PROJECT_DIR / "kb" / "vectordb" / "milvus.db"
    milvus_collection: str = "linear_algebra_kb"

    # 完整历史永久保存在 SQLite；这里只限制每次发送给模型的上下文窗口。
    session_context_max_messages: int = 40
    session_context_max_chars: int = 30000

    @field_validator(
        "embed_model_path", "rerank_model_path", "vlm_model_path", "milvus_db_path"
    )
    @classmethod
    def resolve_project_relative_path(cls, value: Path) -> Path:
        return value if value.is_absolute() else (PROJECT_DIR / value).resolve()

    def require_llm_api_key(self) -> str:
        if not self.llm_api_key:
            raise ConfigurationError("未配置 LLM_API_KEY，无法调用教学模型")
        return self.llm_api_key

    def require_neo4j_password(self) -> str:
        if not self.neo4j_password:
            raise ConfigurationError("未配置 NEO4J_PASSWORD，无法查询知识图谱")
        return self.neo4j_password


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def create_llm_client():
    """延迟创建 OpenAI 兼容客户端，避免导入模块时校验外部配置。"""
    from openai import OpenAI

    settings = get_settings()
    return OpenAI(
        api_key=settings.require_llm_api_key(),
        base_url=settings.llm_base_url,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )
