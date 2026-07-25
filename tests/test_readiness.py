import app.readiness as readiness
from app.config import Settings


def test_missing_runtime_dependency_makes_service_not_ready(monkeypatch, tmp_path):
    original_find_spec = readiness.importlib.util.find_spec

    def fake_find_spec(name):
        if name == "sentence_transformers":
            return None
        return original_find_spec(name)

    monkeypatch.setattr(readiness.importlib.util, "find_spec", fake_find_spec)
    settings = Settings(
        llm_api_key="test-key",
        embed_model_path=tmp_path / "embed",
        rerank_model_path=tmp_path / "rerank",
        milvus_db_path=tmp_path / "milvus.db",
    )

    report = readiness.check_readiness(settings)

    assert report["ready"] is False
    assert report["components"]["runtime_dependencies_present"] is False
    assert "sentence_transformers" in report["details"]["missing_modules"]
