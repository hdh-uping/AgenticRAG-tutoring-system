"""将单个文档的 chunk 增量写入现有 Milvus Lite collection。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")
CHUNKS_DIR = BASE_DIR / "kb" / "data"
MANIFEST_DIR = BASE_DIR / "kb" / "manifests"
DB_FILE = os.getenv("MILVUS_DB_PATH", str(BASE_DIR / "kb" / "vectordb" / "milvus.db"))
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "linear_algebra_kb")
MODEL_PATH = os.getenv(
    "EMBED_MODEL_PATH",
    str(BASE_DIR.parent / "models" / "Qwen" / "Qwen3-Embedding-0.6B"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_standard_artifacts(doc_name: str) -> dict:
    """确认文档已经完成统一的 MinerU → Markdown/content.json → chunk 流程。"""
    doc_dir = BASE_DIR / "kb" / "processed" / doc_name
    required = {
        "source_pdf": BASE_DIR / "kb" / "processed" / f"{doc_name}.pdf",
        "markdown": doc_dir / f"{doc_name}.md",
        "text_markdown": doc_dir / f"{doc_name}_text.md",
        "content_json": doc_dir / f"{doc_name}_content.json",
        "chunks_jsonl": CHUNKS_DIR / f"chunks_{doc_name}.jsonl",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    stale = []
    if not missing:
        source_mtime = max(
            required["markdown"].stat().st_mtime,
            required["text_markdown"].stat().st_mtime,
            required["content_json"].stat().st_mtime,
        )
        if required["chunks_jsonl"].stat().st_mtime < source_mtime:
            stale.append("chunks_jsonl")
    checksum_mismatch = []
    manifest_path = MANIFEST_DIR / f"{doc_name}.json"
    if not missing and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checks = {
            "source_pdf": "source_pdf_sha256",
            "markdown": "markdown_sha256",
            "text_markdown": "text_markdown_sha256",
            "content_json": "content_json_sha256",
            "chunks_jsonl": "chunks_jsonl_sha256",
        }
        checksum_mismatch = [
            artifact for artifact, manifest_key in checks.items()
            if manifest.get(manifest_key) != sha256_file(required[artifact])
        ]
    return {
        "valid": not missing and not stale and not checksum_mismatch,
        "missing": missing,
        "stale": stale,
        "checksum_mismatch": checksum_mismatch,
        "manifest": str(manifest_path) if manifest_path.exists() else None,
        "artifacts": {name: str(path) for name, path in required.items()},
    }


def load_chunks(doc_name: str) -> list[dict]:
    path = CHUNKS_DIR / f"chunks_{doc_name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"未找到文档 chunk: {path}")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ingest_document(doc_name: str, batch_size: int = 16) -> dict:
    artifact_report = validate_standard_artifacts(doc_name)
    if not artifact_report["valid"]:
        problems = []
        if artifact_report["missing"]:
            problems.append(f"缺少: {', '.join(artifact_report['missing'])}")
        if artifact_report["stale"]:
            problems.append(f"已过期: {', '.join(artifact_report['stale'])}")
        if artifact_report["checksum_mismatch"]:
            problems.append(
                f"校验和不一致: {', '.join(artifact_report['checksum_mismatch'])}"
            )
        raise RuntimeError(
            f"文档 {doc_name} 未完成标准入库流程，{'；'.join(problems)}。"
            "禁止写入正式 collection。"
        )
    chunks = load_chunks(doc_name)
    if not chunks:
        raise ValueError(f"文档 {doc_name} 没有可写入的 chunk")

    from pymilvus import MilvusClient
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_PATH, device="cpu")
    embeddings = model.encode(
        [chunk["text"] for chunk in chunks],
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    client = MilvusClient(DB_FILE)
    if not client.has_collection(COLLECTION_NAME):
        raise RuntimeError(f"collection 不存在: {COLLECTION_NAME}；请先执行完整建库脚本")

    records = []
    for chunk, embedding in zip(chunks, embeddings):
        records.append({
            "id": chunk["id"],
            "vector": embedding.tolist(),
            "text": chunk["text"],
            "header_path": chunk.get("header_path", ""),
            "doc_name": doc_name,
            "page_num": chunk.get("page_num", 0),
            "char_count": chunk.get("char_count", 0),
        })

    for start in range(0, len(records), batch_size):
        client.upsert(
            collection_name=COLLECTION_NAME,
            data=records[start:start + batch_size],
        )
    client.flush(COLLECTION_NAME)
    stats = client.get_collection_stats(COLLECTION_NAME)
    return {
        "document": doc_name,
        "upserted": len(records),
        "row_count": int(stats["row_count"]),
        "collection": COLLECTION_NAME,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", required=True, help="文档名，例如：数组")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(ingest_document(args.doc, args.batch_size), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
