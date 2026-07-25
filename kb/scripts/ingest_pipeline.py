"""统一的新文档入库流水线。

数据流：PDF → MinerU → VLM 图片描述 → 结构化切分 → 全量图谱重建 →
校验 → 按文档替换 Milvus + 原子替换 Neo4j 知识图谱。

所有生成步骤先写入 ``kb/.staging``。只有全部校验通过后才提交正式数据；
文件或向量提交失败时会恢复旧版本，Neo4j 图谱使用单事务替换。
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from kb.scripts.validate_graph import validate_graph


KB_DIR = BASE_DIR / "kb"
PROCESSED_DIR = KB_DIR / "processed"
DATA_DIR = KB_DIR / "data"
GRAPH_DIR = KB_DIR / "graph"
MANIFEST_DIR = KB_DIR / "manifests"
STAGING_DIR = KB_DIR / ".staging"
LOCK_PATH = STAGING_DIR / "ingest.lock"

load_dotenv(BASE_DIR / ".env")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class PipelineError(RuntimeError):
    """入库阶段失败；正式库应保持原状。"""


@dataclass(frozen=True)
class StagePaths:
    root: Path
    input_pdf: Path
    processed_root: Path
    doc_dir: Path
    data_dir: Path
    graph_dir: Path
    manifest_path: Path

    @classmethod
    def create(cls, doc_name: str) -> "StagePaths":
        run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        root = STAGING_DIR / f"{doc_name}-{run_id}"
        return cls(
            root=root,
            input_pdf=root / "input" / f"{doc_name}.pdf",
            processed_root=root / "processed",
            doc_dir=root / "processed" / doc_name,
            data_dir=root / "data",
            graph_dir=root / "graph",
            manifest_path=root / "manifest" / f"{doc_name}.json",
        )


def validate_doc_name(doc_name: str) -> str:
    doc_name = doc_name.strip()
    if not doc_name or not re.fullmatch(r"[\w\u4e00-\u9fff ()（）·-]+", doc_name):
        raise PipelineError(
            "文档名只能包含中英文、数字、空格、括号、连字符或间隔号"
        )
    if len(doc_name) > 80:
        raise PipelineError("文档名不能超过 80 个字符")
    return doc_name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def ingestion_lock():
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("w", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PipelineError("已有另一项文档入库任务正在运行") from exc
        yield


def run_step(label: str, command: list[str]) -> None:
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    result = subprocess.run(command, cwd=BASE_DIR, check=False)
    if result.returncode != 0:
        raise PipelineError(f"{label}失败（退出码 {result.returncode}）")


def verify_llm_configuration() -> None:
    """在批量抽取前用一次极小请求验证密钥、地址和模型名。"""
    from openai import OpenAI

    api_key = os.getenv("LLM_API_KEY", "")
    base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
    model = os.getenv("LLM_MODEL", "deepseek-v4-flash")
    if not api_key:
        raise PipelineError("未配置 LLM_API_KEY")
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
        max_retries=0,
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "只回复OK"}],
            temperature=0,
            max_tokens=2,
            extra_body={"thinking": {"type": "disabled"}},
        )
    except Exception as exc:
        raise PipelineError(
            f"LLM 预检失败，请核对 LLM_API_KEY/LLM_BASE_URL/LLM_MODEL: {exc}"
        ) from exc
    if not response.choices:
        raise PipelineError("LLM 预检未返回有效结果")
    print(f"LLM 预检通过: {base_url} / {model}")


def prepare_document(
    source_pdf: Path,
    doc_name: str,
    stage: StagePaths,
    reuse_processed: bool,
) -> None:
    stage.input_pdf.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_pdf, stage.input_pdf)

    if reuse_processed:
        existing = PROCESSED_DIR / doc_name
        if not existing.is_dir():
            raise PipelineError(f"没有可复用的标准解析目录: {existing}")
        shutil.copytree(existing, stage.doc_dir)
        print(f"复用已校验的解析产物: {existing}")
        return

    run_step(
        "1/4 MinerU 标准解析",
        [
            sys.executable,
            str(KB_DIR / "scripts" / "parse_pdf.py"),
            "--pdf",
            str(stage.input_pdf),
            "--output-dir",
            str(stage.processed_root),
        ],
    )
    run_step(
        "2/4 VLM 图片语义转写",
        [
            sys.executable,
            str(KB_DIR / "scripts" / "describe_images.py"),
            "--md-file",
            str(stage.doc_dir / f"{doc_name}.md"),
        ],
    )


def validate_processed_artifacts(stage: StagePaths, doc_name: str) -> dict:
    paths = {
        "pdf": stage.input_pdf,
        "markdown": stage.doc_dir / f"{doc_name}.md",
        "text_markdown": stage.doc_dir / f"{doc_name}_text.md",
        "content_json": stage.doc_dir / f"{doc_name}_content.json",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise PipelineError(f"标准解析产物缺失: {', '.join(missing)}")

    content = json.loads(paths["content_json"].read_text(encoding="utf-8"))
    if not isinstance(content, list) or not content:
        raise PipelineError("MinerU content.json 为空或格式错误")
    text = paths["text_markdown"].read_text(encoding="utf-8").strip()
    if not text:
        raise PipelineError("VLM 处理后的 _text.md 为空")
    if re.search(r"!\[[^]]*\]\(images/[^)]+\)", text):
        raise PipelineError("_text.md 中仍存在未处理的正文图片引用")
    return {"content_elements": len(content), "text_chars": len(text)}


def build_chunks(stage: StagePaths, doc_name: str) -> list[dict]:
    run_step(
        "3/4 结构化切分",
        [
            sys.executable,
            str(KB_DIR / "scripts" / "chunk_text.py"),
            "--text-file",
            str(stage.doc_dir / f"{doc_name}_text.md"),
            "--output-dir",
            str(stage.data_dir),
        ],
    )
    json_path = stage.data_dir / f"chunks_{doc_name}.json"
    jsonl_path = stage.data_dir / f"chunks_{doc_name}.jsonl"
    chunks = json.loads(json_path.read_text(encoding="utf-8"))
    jsonl_chunks = [
        json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not chunks or chunks != jsonl_chunks:
        raise PipelineError("chunk JSON 与 JSONL 不一致或没有有效 chunk")
    ids = [chunk.get("id") for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise PipelineError("chunk ID 重复")
    for chunk in chunks:
        if not str(chunk.get("id", "")).startswith(f"{doc_name}_chunk_"):
            raise PipelineError(f"chunk ID 不属于当前文档: {chunk.get('id')}")
        if not chunk.get("text") or chunk.get("page_num", 0) <= 0:
            raise PipelineError(f"chunk 内容为空或页码无效: {chunk.get('id')}")
        if len(chunk["id"]) > 32:
            raise PipelineError(f"chunk ID 超过 Milvus 32 字符限制: {chunk['id']}")
    return chunks


def graph_content_files(stage: StagePaths, doc_name: str) -> list[Path]:
    official = [
        path for path in sorted(PROCESSED_DIR.glob("*/*_content.json"))
        if path.stem.removesuffix("_content") != doc_name
    ]
    staged = stage.doc_dir / f"{doc_name}_content.json"
    files = official + [staged]
    if not files:
        raise PipelineError("没有可用于构建图谱的 content.json")
    return files


def build_and_validate_graph(
    stage: StagePaths,
    content_files: list[Path],
) -> tuple[list[dict], list[dict], dict]:
    command = [
        sys.executable,
        str(KB_DIR / "scripts" / "build_graph.py"),
        "--output-dir",
        str(stage.graph_dir),
        "--no-import",
        "--cache-dir",
        str(KB_DIR / ".cache" / "graph"),
    ]
    for content_file in content_files:
        command.extend(["--content-file", str(content_file)])
    run_step("4/4 全量图谱抽取（暂存）", command)

    entities_path = stage.graph_dir / "entities.json"
    relations_path = stage.graph_dir / "relations.json"
    report = validate_graph(entities_path, relations_path, content_files)
    if not report["valid"] or not report["provenance_consistent"]:
        raise PipelineError(
            "暂存图谱校验失败: "
            + json.dumps(
                {"errors": report["errors"], "warnings": report["warnings"]},
                ensure_ascii=False,
            )
        )
    entities = json.loads(entities_path.read_text(encoding="utf-8"))
    relations = json.loads(relations_path.read_text(encoding="utf-8"))
    return entities, relations, report


def prepare_vector_records(doc_name: str, chunks: list[dict]) -> list[dict]:
    from sentence_transformers import SentenceTransformer

    model_path = os.getenv(
        "EMBED_MODEL_PATH",
        str(BASE_DIR.parent / "models" / "Qwen" / "Qwen3-Embedding-0.6B"),
    )
    if not Path(model_path).is_dir():
        raise PipelineError(f"Embedding 模型目录不存在: {model_path}")
    print(f"\n生成 {len(chunks)} 个暂存向量...")
    model = SentenceTransformer(model_path, device="cpu")
    embeddings = model.encode(
        [chunk["text"] for chunk in chunks],
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return [
        {
            "id": chunk["id"],
            "vector": embedding.tolist(),
            "text": chunk["text"],
            "header_path": chunk.get("header_path", ""),
            "doc_name": doc_name,
            "page_num": chunk.get("page_num", 0),
            "char_count": chunk.get("char_count", 0),
        }
        for chunk, embedding in zip(chunks, embeddings)
    ]


def milvus_filter(doc_name: str) -> str:
    return f'doc_name == "{doc_name}"'


def replace_document_vectors(doc_name: str, records: list[dict]) -> dict:
    from pymilvus import MilvusClient

    db_file = os.getenv("MILVUS_DB_PATH", str(KB_DIR / "vectordb" / "milvus.db"))
    collection = os.getenv("MILVUS_COLLECTION", "linear_algebra_kb")
    client = MilvusClient(db_file)
    if not client.has_collection(collection):
        raise PipelineError(f"Milvus collection 不存在: {collection}")
    client.load_collection(collection)

    old_records = client.query(
        collection_name=collection,
        filter=milvus_filter(doc_name),
        output_fields=[
            "id", "vector", "text", "header_path", "doc_name", "page_num", "char_count"
        ],
        limit=16384,
    )
    changed = False
    try:
        client.delete(collection_name=collection, filter=milvus_filter(doc_name))
        changed = True
        for start in range(0, len(records), 16):
            client.upsert(collection_name=collection, data=records[start:start + 16])
        client.flush(collection)
    except Exception:
        if changed:
            client.delete(collection_name=collection, filter=milvus_filter(doc_name))
            for start in range(0, len(old_records), 16):
                client.upsert(collection_name=collection, data=old_records[start:start + 16])
            client.flush(collection)
        raise

    stats = client.get_collection_stats(collection)
    return {
        "old_rows": len(old_records),
        "new_rows": len(records),
        "collection_rows": int(stats["row_count"]),
        "old_records": old_records,
        "db_file": db_file,
        "collection": collection,
    }


def rollback_vectors(doc_name: str, vector_state: dict) -> None:
    from pymilvus import MilvusClient

    client = MilvusClient(vector_state["db_file"])
    collection = vector_state["collection"]
    client.load_collection(collection)
    client.delete(collection_name=collection, filter=milvus_filter(doc_name))
    old_records = vector_state["old_records"]
    for start in range(0, len(old_records), 16):
        client.upsert(collection_name=collection, data=old_records[start:start + 16])
    client.flush(collection)


def verify_neo4j(skip_neo4j: bool) -> bool:
    password = os.getenv("NEO4J_PASSWORD", "")
    if skip_neo4j or not password:
        print("Neo4j 未启用；正式图谱以 kb/graph JSON 为准")
        return False
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), password),
    )
    try:
        driver.verify_connectivity()
    finally:
        driver.close()
    return True


def verify_commit_targets(skip_neo4j: bool) -> bool:
    """在付费图谱抽取前确认向量库、本地模型和可选 Neo4j 可提交。"""
    from pymilvus import MilvusClient

    model_path = Path(os.getenv(
        "EMBED_MODEL_PATH",
        str(BASE_DIR.parent / "models" / "Qwen" / "Qwen3-Embedding-0.6B"),
    ))
    if not model_path.is_dir():
        raise PipelineError(f"Embedding 模型目录不存在: {model_path}")
    db_file = os.getenv("MILVUS_DB_PATH", str(KB_DIR / "vectordb" / "milvus.db"))
    collection = os.getenv("MILVUS_COLLECTION", "linear_algebra_kb")
    client = MilvusClient(db_file)
    try:
        if not client.has_collection(collection):
            raise PipelineError(f"Milvus collection 不存在: {collection}")
        client.load_collection(collection)
    finally:
        client.close()
    return verify_neo4j(skip_neo4j)


def build_manifest(
    stage: StagePaths,
    doc_name: str,
    parse_stats: dict,
    chunks: list[dict],
    graph_report: dict,
) -> dict:
    manifest = {
        "schema_version": 1,
        "pipeline": "kb.scripts.ingest_pipeline",
        "document": doc_name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_pdf_sha256": sha256_file(stage.input_pdf),
        "markdown_sha256": sha256_file(stage.doc_dir / f"{doc_name}.md"),
        "text_markdown_sha256": sha256_file(stage.doc_dir / f"{doc_name}_text.md"),
        "content_json_sha256": sha256_file(stage.doc_dir / f"{doc_name}_content.json"),
        "chunks_jsonl_sha256": sha256_file(stage.data_dir / f"chunks_{doc_name}.jsonl"),
        "content_elements": parse_stats["content_elements"],
        "text_chars": parse_stats["text_chars"],
        "chunk_count": len(chunks),
        "pages": sorted({chunk["page_num"] for chunk in chunks}),
        "graph_counts": graph_report["counts"],
    }
    stage.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    stage.manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def artifact_pairs(stage: StagePaths, doc_name: str) -> list[tuple[Path, Path]]:
    return [
        (stage.input_pdf, PROCESSED_DIR / f"{doc_name}.pdf"),
        (stage.doc_dir, PROCESSED_DIR / doc_name),
        (stage.data_dir / f"chunks_{doc_name}.json", DATA_DIR / f"chunks_{doc_name}.json"),
        (stage.data_dir / f"chunks_{doc_name}.jsonl", DATA_DIR / f"chunks_{doc_name}.jsonl"),
        (stage.data_dir / f"chunks_{doc_name}_txt", DATA_DIR / f"chunks_{doc_name}_txt"),
        (stage.graph_dir / "entities.json", GRAPH_DIR / "entities.json"),
        (stage.graph_dir / "relations.json", GRAPH_DIR / "relations.json"),
        (stage.manifest_path, MANIFEST_DIR / f"{doc_name}.json"),
    ]


def promote_files(stage: StagePaths, doc_name: str) -> tuple[list[Path], Path]:
    backup_root = stage.root / "backup"
    promoted = []
    try:
        for source, target in artifact_pairs(stage, doc_name):
            if not source.exists():
                raise PipelineError(f"待提交产物不存在: {source}")
            backup = backup_root / target.relative_to(BASE_DIR)
            backup.parent.mkdir(parents=True, exist_ok=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                shutil.move(str(target), str(backup))
            shutil.move(str(source), str(target))
            promoted.append(target)
    except Exception:
        rollback_files(promoted, backup_root)
        raise
    return promoted, backup_root


def rollback_files(promoted: list[Path], backup_root: Path) -> None:
    for target in reversed(promoted):
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
        backup = backup_root / target.relative_to(BASE_DIR)
        if backup.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(backup), str(target))


def commit(
    stage: StagePaths,
    doc_name: str,
    vector_records: list[dict],
    entities: list[dict],
    relations: list[dict],
    use_neo4j: bool,
) -> dict:
    promoted: list[Path] = []
    backup_root = stage.root / "backup"
    vector_state = None
    try:
        promoted, backup_root = promote_files(stage, doc_name)
        vector_state = replace_document_vectors(doc_name, vector_records)
        if use_neo4j:
            from kb.scripts.build_graph import import_to_neo4j

            import_to_neo4j(entities, relations)
        return {key: value for key, value in vector_state.items() if key != "old_records"}
    except Exception:
        if vector_state is not None:
            rollback_vectors(doc_name, vector_state)
        if promoted:
            rollback_files(promoted, backup_root)
        raise


def run_pipeline(args: argparse.Namespace) -> dict:
    source_pdf = args.pdf.resolve()
    if not source_pdf.is_file() or source_pdf.suffix.lower() != ".pdf":
        raise PipelineError(f"PDF 不存在或格式不正确: {source_pdf}")
    doc_name = validate_doc_name(args.doc_name or source_pdf.stem)
    stage = StagePaths.create(doc_name)
    stage.root.mkdir(parents=True, exist_ok=False)

    try:
        prepare_document(source_pdf, doc_name, stage, args.reuse_processed)
        parse_stats = validate_processed_artifacts(stage, doc_name)
        verify_llm_configuration()
        chunks = build_chunks(stage, doc_name)
        content_files = graph_content_files(stage, doc_name)
        entities, relations, graph_report = build_and_validate_graph(stage, content_files)
        manifest = build_manifest(stage, doc_name, parse_stats, chunks, graph_report)

        if args.dry_run:
            return {
                "status": "validated_not_committed",
                "document": doc_name,
                "staging_dir": str(stage.root),
                "manifest": manifest,
            }

        vector_records = prepare_vector_records(doc_name, chunks)
        # gRPC 客户端放在 tokenizer/torch 完成后创建，避免 fork 线程冲突。
        use_neo4j = verify_commit_targets(args.skip_neo4j)
        vector_report = commit(
            stage, doc_name, vector_records, entities, relations, use_neo4j
        )
        return {
            "status": "committed",
            "document": doc_name,
            "staging_dir": str(stage.root),
            "manifest": manifest,
            "vector": vector_report,
            "neo4j_updated": use_neo4j,
        }
    except Exception:
        print(f"\n入库失败，暂存目录保留用于排查: {stage.root}")
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="标准化、可回滚的新文档入库")
    parser.add_argument("--pdf", type=Path, required=True, help="待入库 PDF")
    parser.add_argument("--doc-name", help="文档名；默认使用 PDF 文件名")
    parser.add_argument(
        "--reuse-processed",
        action="store_true",
        help="复用 kb/processed/{doc} 的 MinerU/VLM 标准产物",
    )
    parser.add_argument("--dry-run", action="store_true", help="只暂存和校验，不提交")
    parser.add_argument("--skip-neo4j", action="store_true", help="只更新 JSON 图谱")
    parser.add_argument("--keep-staging", action="store_true", help="成功后保留暂存目录")
    args = parser.parse_args()

    with ingestion_lock():
        report = run_pipeline(args)
    print("\n" + json.dumps(report, ensure_ascii=False, indent=2))

    if report["status"] == "committed" and not args.keep_staging:
        staging_dir = report.get("staging_dir")
        if staging_dir:
            shutil.rmtree(staging_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
