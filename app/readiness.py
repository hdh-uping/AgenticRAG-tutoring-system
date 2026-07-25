"""不调用付费外部服务的后端就绪检查。"""
import importlib.util
import json
from functools import lru_cache
from pathlib import Path

from app.config import Settings, get_settings


REQUIRED_RUNTIME_MODULES = (
    "openai",
    "sentence_transformers",
    "transformers",
    "torch",
    "pymilvus",
    "jieba",
    "rank_bm25",
)


def _model_assets_present(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").exists():
        return False
    return any(path.glob("*.safetensors")) or any(path.glob("*.bin"))


def _local_graph_valid(project_dir: Path) -> bool:
    graph_dir = project_dir / "kb" / "graph"
    try:
        entities = json.loads((graph_dir / "entities.json").read_text(encoding="utf-8"))
        relations = json.loads((graph_dir / "relations.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(entities, list) and bool(entities) and isinstance(relations, list)


@lru_cache(maxsize=4)
def _probe_milvus(
    db_path: str,
    collection_name: str,
    path_mtime_ns: int,
) -> tuple[bool, int, str]:
    """实际打开一次 collection；路径时间戳变化时自动重新检查。"""
    del path_mtime_ns
    client = None
    try:
        from pymilvus import MilvusClient

        client = MilvusClient(db_path)
        collections = client.list_collections()
        if collection_name not in collections:
            return False, 0, f"collection 不存在: {collection_name}"
        client.load_collection(collection_name)
        stats = client.get_collection_stats(collection_name)
        row_count = int(stats.get("row_count", 0))
        return row_count > 0, row_count, "" if row_count > 0 else "collection 为空"
    except Exception as exc:
        return False, 0, f"{type(exc).__name__}: {exc}"
    finally:
        if client is not None:
            client.close()


def check_readiness(settings: Settings | None = None) -> dict:
    """检查依赖、模型资产、图谱和实际 Milvus collection。"""
    settings = settings or get_settings()
    missing_modules = [
        name for name in REQUIRED_RUNTIME_MODULES
        if importlib.util.find_spec(name) is None
    ]
    runtime_ok = not missing_modules
    embed_ok = _model_assets_present(settings.embed_model_path)
    rerank_ok = _model_assets_present(settings.rerank_model_path)
    graph_ok = _local_graph_valid(Path(__file__).resolve().parent.parent)

    milvus_collection_ok = False
    milvus_row_count = 0
    milvus_error = ""
    if runtime_ok and settings.milvus_db_path.exists():
        milvus_collection_ok, milvus_row_count, milvus_error = _probe_milvus(
            str(settings.milvus_db_path),
            settings.milvus_collection,
            settings.milvus_db_path.stat().st_mtime_ns,
        )

    components = {
        "llm_configured": bool(settings.llm_api_key),
        "runtime_dependencies_present": runtime_ok,
        "embed_model_present": embed_ok,
        "rerank_model_present": rerank_ok,
        # Milvus Lite 2.x 常用单文件，3.x 可把同一路径实现为目录。
        "milvus_db_present": settings.milvus_db_path.exists(),
        "milvus_collection_available": milvus_collection_ok,
        "graph_available": graph_ok,
        "neo4j_configured": bool(settings.neo4j_password),
    }
    required = (
        "llm_configured",
        "runtime_dependencies_present",
        "embed_model_present",
        "rerank_model_present",
        "milvus_db_present",
        "milvus_collection_available",
        "graph_available",
    )
    return {
        "ready": all(components[name] for name in required),
        "components": components,
        "details": {
            "missing_modules": missing_modules,
            "milvus_collection": settings.milvus_collection,
            "milvus_row_count": milvus_row_count,
            "milvus_error": milvus_error,
        },
    }
