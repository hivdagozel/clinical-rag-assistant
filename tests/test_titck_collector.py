from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.collect_titck_documents as collector_module


class FakeResponse:
    def __init__(self, *, payload=None, content=b"", content_type="application/json", text=""):
        self._payload = payload
        self.content = content
        self.headers = {"Content-Type": content_type}
        self.text = text
        self.status_code = 200

    def json(self):
        return self._payload


@pytest.fixture
def collector(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    metadata = tmp_path / "metadata" / "documents.json"
    metadata.parent.mkdir()
    metadata.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(collector_module, "METADATA_FILE", metadata)
    monkeypatch.setattr(collector_module, "KT_PDF_DIR", tmp_path / "kt")
    monkeypatch.setattr(collector_module, "KUB_PDF_DIR", tmp_path / "kub")
    monkeypatch.setattr(collector_module, "QUARANTINE_DIR", tmp_path / "quarantine")
    monkeypatch.setattr(collector_module, "REJECTED_PDF_DIR", tmp_path / "rejected")
    monkeypatch.setattr(collector_module, "MANUAL_REVIEW_DIR", tmp_path / "manual_review")
    return collector_module.TITCKCollector("kt", 20, 1.5, False)


def test_fetch_page_uses_datatables_pagination_and_total(collector, monkeypatch):
    collector.csrf_token = "token"
    captured = {}

    def fake_request(method, url, **kwargs):
        captured.update(kwargs["data"])
        return FakeResponse(payload={"data": [{"name": "A"}], "recordsFiltered": 4321})

    monkeypatch.setattr(collector, "_request", fake_request)
    records, total = collector.fetch_page(200, length=100)
    assert records == [{"name": "A"}]
    assert total == 4321
    assert collector.report.total_records == 4321
    assert captured["start"] == "200"
    assert captured["length"] == "100"
    assert captured["search[value]"] == ""


def test_non_pdf_content_type_is_rejected_without_writing(collector, monkeypatch):
    monkeypatch.setattr(collector, "_request", lambda *a, **k: FakeResponse(content=b"not pdf", content_type="text/plain"))
    record = {"name": "TESTMED", "documentPathKt": '<a href="/docs/test.pdf">KT</a>'}
    collector.process_record(record)
    assert collector.report.rejected == 1
    assert collector.report.downloaded_pdf == 0
    assert not list(collector.accepted_dir.glob("*.pdf"))
    assert collector.report.failed_records[0]["reason"].startswith("Geçersiz Content-Type")


def test_html_instead_of_pdf_triggers_safety_stop(collector, monkeypatch):
    monkeypatch.setattr(collector, "_request", lambda *a, **k: FakeResponse(content=b"<html>", content_type="text/html"))
    record = {"name": "TESTMED", "documentPathKt": '<a href="/docs/test.pdf">KT</a>'}
    with pytest.raises(collector_module.SafetyStop, match="HTML"):
        collector.process_record(record)


def test_accepted_document_has_required_metadata(collector, monkeypatch):
    content = b"%PDF fake-for-unit-test"
    monkeypatch.setattr(collector, "_request", lambda *a, **k: FakeResponse(content=content, content_type="application/pdf"))
    monkeypatch.setattr(collector, "_verify_pdf", lambda *_: collector_module.ValidationResult(
        "accepted", "accepted", 7, 1000, ["kullanma_talimati"], 13, True, "KT"
    ))
    record = {
        "name": "TESTMED 500 MG TABLET",
        "approvalDate": "2026-07-01",
        "documentPathKt": '<a href="/docs/test.pdf">KT</a>',
    }
    collector.process_record(record)
    item = next(iter(collector.metadata.values()))
    required = {"drug_name", "document_type", "page_count", "approval_date", "source_url", "sha256", "download_time"}
    assert required <= set(item)
    assert item["document_type"] == "KT"
    assert item["page_count"] == 7
    assert item["approval_date"] == "2026-07-01"
    assert collector.report.accepted == 1


def test_checkpoint_resume_preserves_page_and_urls(collector):
    collector.processed_urls.add("https://www.titck.gov.tr/docs/test.pdf")
    collector._save_state(300)
    saved = json.loads(collector.checkpoint_file.read_text(encoding="utf-8"))
    assert saved["start"] == 300
    assert saved["processed_urls"] == ["https://www.titck.gov.tr/docs/test.pdf"]


def test_validator_accepts_layout_spaces_in_real_kt_headings(collector, monkeypatch):
    text = (
        "KULLANMA TAL İMATI TESTMED 500 MG TABLET "
        "Bu ilacı kullanmadan önce dikkat edilmesi gerekenler. "
        "Nasıl kullanılır? Olası yan etkiler nelerdir? İlacın saklanması. "
    ) * 4

    class Page:
        def extract_text(self):
            return text

    class Reader:
        pages = [Page(), Page()]

    monkeypatch.setattr(collector_module.pypdf, "PdfReader", lambda *_: Reader())
    result = collector._verify_pdf(b"%PDF unit", "TESTMED 500 MG TABLET")
    assert result.decision == "accepted"
    assert result.detected_type == "KT"
    assert result.score >= 10


def test_scanned_pdf_goes_to_manual_review(collector, monkeypatch):
    class Page:
        def extract_text(self):
            return ""

    class Reader:
        pages = [Page(), Page()]

    monkeypatch.setattr(collector_module.pypdf, "PdfReader", lambda *_: Reader())
    result = collector._verify_pdf(b"%PDF unit", "TESTMED")
    assert result.decision == "manual_review"
    assert "OCR" in result.reason


def test_targeted_mode_uses_separate_checkpoint_and_skips_non_targets(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata" / "documents.json"
    metadata.parent.mkdir()
    metadata.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(collector_module, "METADATA_FILE", metadata)
    monkeypatch.setattr(collector_module, "KT_PDF_DIR", tmp_path / "kt")
    monkeypatch.setattr(collector_module, "KUB_PDF_DIR", tmp_path / "kub")
    monkeypatch.setattr(collector_module, "QUARANTINE_DIR", tmp_path / "quarantine")
    monkeypatch.setattr(collector_module, "REJECTED_PDF_DIR", tmp_path / "rejected")
    monkeypatch.setattr(collector_module, "MANUAL_REVIEW_DIR", tmp_path / "manual_review")
    targeted = collector_module.TITCKCollector(
        "kt", 100, 1.5, True, selection_mode="targeted",
        categories=("diabetes",), max_products_per_ingredient_form=5,
    )
    assert targeted.checkpoint_file.name == "collector_kt_targeted_checkpoint.json"
    targeted.process_record({
        "name": "HEDEF DIŞI ÜRÜN",
        "activeIngredient": "başka madde",
        "documentPathKt": '<a href="/docs/test.pdf">KT</a>',
    })
    assert targeted.report.download_attempts == 0
