"""校验问题集、知识块和图谱概念之间的引用完整性。"""
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


def validate_questions(dataset_path: Path = BASE_DIR / "eval" / "questions.json") -> dict:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = dataset.get("cases", [])
    errors = []

    case_ids = [case.get("id") for case in cases]
    duplicates = sorted({case_id for case_id in case_ids if case_ids.count(case_id) > 1})
    if duplicates:
        errors.append(f"重复问题ID: {', '.join(duplicates)}")

    chunks = {}
    for path in (BASE_DIR / "kb" / "data").glob("chunks_*.jsonl"):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                chunks[chunk["id"]] = chunk
    chunk_ids = set(chunks)

    entities = json.loads((BASE_DIR / "kb" / "graph" / "entities.json").read_text(encoding="utf-8"))
    entity_names = {entity["name"] for entity in entities}
    source_pdfs = {path.name for path in (BASE_DIR / "kb" / "processed").glob("*.pdf")}

    for case in cases:
        case_id = case.get("id", "<missing>")
        if case.get("origin") not in {"original", "derived"}:
            errors.append(f"{case_id}: origin非法")
        if not case.get("question") or not case.get("expected_answer"):
            errors.append(f"{case_id}: 缺少问题或答案")
        missing_chunks = sorted(set(case.get("expected_chunk_ids", [])) - chunk_ids)
        if missing_chunks:
            errors.append(f"{case_id}: chunk不存在 {missing_chunks}")
        evidence_text = "\n".join(
            chunks[chunk_id].get("text", "")
            for chunk_id in case.get("expected_chunk_ids", [])
            if chunk_id in chunks
        )
        missing_evidence = [
            term for term in case.get("expected_evidence_terms", [])
            if term not in evidence_text
        ]
        if missing_evidence:
            errors.append(f"{case_id}: 引用chunk缺少证据 {missing_evidence}")
        missing_concepts = sorted(set(case.get("expected_concepts", [])) - entity_names)
        if missing_concepts:
            errors.append(f"{case_id}: 图谱概念不存在 {missing_concepts}")
        if case.get("origin") == "original" and case.get("source_document") not in source_pdfs:
            errors.append(f"{case_id}: 来源PDF不存在")
        if case.get("origin") == "derived" and case.get("based_on") not in case_ids:
            errors.append(f"{case_id}: based_on不存在")

    return {
        "valid": not errors,
        "errors": errors,
        "counts": {
            "total": len(cases),
            "original": sum(case.get("origin") == "original" for case in cases),
            "derived": sum(case.get("origin") == "derived" for case in cases),
        },
    }


def main() -> int:
    report = validate_questions()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
