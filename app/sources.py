"""教材 chunk 的只读查询。"""
import json
from functools import lru_cache
from pathlib import Path


CHUNKS_DIR = Path(__file__).resolve().parent.parent / "kb" / "data"


@lru_cache(maxsize=1)
def load_sources() -> dict[str, dict]:
    sources = {}
    for path in sorted(CHUNKS_DIR.glob("chunks_*.jsonl")):
        document = path.stem.removeprefix("chunks_")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                source = dict(chunk)
                source["document"] = document
                sources[source["id"]] = source
    return sources


def get_source(chunk_id: str) -> dict | None:
    return load_sources().get(chunk_id)
