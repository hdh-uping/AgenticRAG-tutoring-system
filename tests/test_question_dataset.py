import json
from pathlib import Path

from eval.validate_questions import validate_questions


def test_question_dataset_references_current_kb():
    report = validate_questions()
    assert report["valid"], report["errors"]
    assert report["counts"] == {"total": 33, "original": 18, "derived": 15}


def test_question_dataset_rejects_chunks_without_required_evidence(tmp_path):
    dataset_path = Path(__file__).resolve().parents[1] / "eval" / "questions.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset["cases"][0]["expected_evidence_terms"] = ["不存在的证据词"]
    invalid_path = tmp_path / "questions.json"
    invalid_path.write_text(
        json.dumps(dataset, ensure_ascii=False), encoding="utf-8"
    )

    report = validate_questions(invalid_path)

    assert report["valid"] is False
    assert any("引用chunk缺少证据" in error for error in report["errors"])
