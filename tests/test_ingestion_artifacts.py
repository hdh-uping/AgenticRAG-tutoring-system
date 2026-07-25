from kb.scripts.ingest_document import validate_standard_artifacts


def test_existing_mineru_document_has_standard_artifacts():
    report = validate_standard_artifacts("线性表")
    assert report["valid"], report["missing"]


def test_array_has_current_standard_artifacts_after_pipeline_commit():
    report = validate_standard_artifacts("数组")
    assert report["valid"] is True
    assert report["missing"] == []
    assert report["stale"] == []
    assert report["checksum_mismatch"] == []
    assert report["manifest"].endswith("kb/manifests/数组.json")
