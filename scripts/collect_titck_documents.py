"""Resumable, defensive TİTCK KT/KÜB collector and pilot validator.

Uses only the public GET -> session/CSRF -> DataTables POST flow. It never
bypasses CAPTCHA, authentication, TLS verification, or access controls.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import random
import re
import shutil
import sys
import time
import unicodedata
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import pypdf
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import KT_PDF_DIR, KUB_PDF_DIR, METADATA_FILE, QUARANTINE_DIR, REJECTED_PDF_DIR
from src.collection_targets import TargetMatch, TargetSelector, canonical_product_fields, normalize as target_normalize

logger = logging.getLogger("titck_collector")
BASE_URL = "https://www.titck.gov.tr"
KUBKT_URL = f"{BASE_URL}/kubkt"
AJAX_URL = f"{BASE_URL}/getkubktviewdatatable"
ALLOWED_PDF_HOSTS = {"www.titck.gov.tr", "titck.gov.tr"}
CAPTCHA_MARKERS = ("captcha", "g-recaptcha", "hcaptcha", "giris yap", "login")
COLLECTOR_VERSION = "2.0"
VALIDATOR_VERSION = "2.0"
TARGETS_FILE = Path(__file__).resolve().parent.parent / "config" / "medicine_collection_targets.yaml"
MANUAL_REVIEW_DIR = METADATA_FILE.parent.parent / "manual_review"
UNRELATED_MARKERS = (
    "sikca sorulan sorular", "duyuru", "basvuru formu", "yonetmelik",
    "mevzuat", "kilavuz", "genel kurul", "faaliyet raporu",
)


class SafetyStop(RuntimeError):
    pass


def normalized(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value)).casefold().replace("ı", "i")
    return " ".join("".join(ch for ch in value if not unicodedata.combining(ch)).split())


def atomic_json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


@dataclass
class Report:
    requested_type: str
    requested_limit: int
    total_records: int = 0
    scanned_records: int = 0
    download_attempts: int = 0
    downloaded_pdf: int = 0
    kt_links_found: int = 0
    accepted: int = 0
    rejected: int = 0
    duplicates: int = 0
    http_errors: int = 0
    manual_review: int = 0
    ocr_required: int = 0
    validation_errors: int = 0
    product_match_errors: int = 0
    total_chunks: int = 0
    total_pdfs: int = 0
    unique_medicines: int = 0
    manifest_valid: bool | None = None
    faiss_status: str = "not_run"
    total_duration_seconds: float = 0.0
    average_record_seconds: float = 0.0
    eta_seconds: float | None = None
    checkpoint_offset: int = 0
    total_accepted_kt: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    failed_records: list[dict[str, str]] = field(default_factory=list)
    rag_tests: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str | None = None
    selection_mode: str = "all"
    category_stats: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class ValidationResult:
    decision: str
    reason: str
    page_count: int
    character_count: int
    detected_headings: list[str]
    score: int
    product_match: bool
    detected_type: str | None


class TITCKCollector:
    def __init__(self, document_type: str, limit: int, delay: float, resume: bool,
                 batch_size: int = 100, retry_rejected: bool = False,
                 from_offset: int | None = None, selection_mode: str = "all",
                 categories: tuple[str, ...] = (), active_ingredients: tuple[str, ...] = (),
                 atc_prefixes: tuple[str, ...] = (), product_list: tuple[str, ...] = (),
                 max_products_per_ingredient_form: int = 5):
        self.document_type = document_type.upper()
        self.limit = limit
        self.delay = max(delay, 1.0)
        self.resume = resume
        self.batch_size = batch_size
        self.retry_rejected = retry_rejected
        self.from_offset = from_offset
        self.selection_mode = selection_mode
        self.requested_categories = tuple(categories)
        self.requested_active_ingredients = tuple(target_normalize(item) for item in active_ingredients)
        self.requested_atc_prefixes = tuple(target_normalize(item).replace(" ", "") for item in atc_prefixes)
        self.requested_products = tuple(target_normalize(item) for item in product_list)
        self.max_products_per_ingredient_form = max_products_per_ingredient_form
        self.target_selector = TargetSelector(TARGETS_FILE, categories) if selection_mode == "targeted" else None
        self.group_counts: Counter[str] = Counter()
        self.accepted_dir = KT_PDF_DIR if self.document_type == "KT" else KUB_PDF_DIR
        mode_suffix = "" if selection_mode == "all" else f"_{selection_mode.replace('-', '_')}"
        self.checkpoint_file = METADATA_FILE.parent / f"collector_{self.document_type.lower()}{mode_suffix}_checkpoint.json"
        self.report_file = METADATA_FILE.parent / f"collector_{self.document_type.lower()}{mode_suffix}_report.json"
        self.session = requests.Session()
        retry = Retry(total=3, connect=3, read=3, backoff_factor=1.0,
                      status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET", "POST"))
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.headers = {
            "User-Agent": "Medical-RAG-Research-Collector/1.0 (+controlled public document retrieval)",
            "Accept-Language": "tr-TR,tr;q=0.9", "Origin": BASE_URL, "Referer": KUBKT_URL,
        }
        self.csrf_token: str | None = None
        self.metadata = self._load_json(METADATA_FILE, default={})
        if not isinstance(self.metadata, dict):
            raise SafetyStop("Metadata şeması bozuk; nesne bekleniyordu.")
        self.checkpoint = self._load_checkpoint()
        self.processed_urls: set[str] = set(self.checkpoint.get("processed_urls", []))
        self.rejected_records: dict[str, str] = dict(self.checkpoint.get("rejected_records", {}))
        self.retry_counts: dict[str, int] = dict(self.checkpoint.get("retry_counts", {}))
        self.known_hashes = {item.get("sha256") for item in self.metadata.values()
                             if isinstance(item, dict) and item.get("sha256")}
        self.report = Report(self.document_type, limit)
        self.report.selection_mode = selection_mode
        if self.target_selector:
            self.report.category_stats = {
                name: {
                    "quota": int(definition.get("quota", 0)), "candidates": 0,
                    "downloaded": 0, "accepted": 0, "rejected": 0,
                    "manual_review": 0, "duplicate": 0,
                    "unique_ingredients": [], "unique_products": [],
                }
                for name, definition in self.target_selector.categories.items()
            }
        self.rejection_counts: Counter[str] = Counter()
        self.recent_decisions: deque[tuple[str, str]] = deque(maxlen=100)
        self.started_at = time.monotonic()
        self.consecutive_http_errors = 0
        for directory in (self.accepted_dir, QUARANTINE_DIR, REJECTED_PDF_DIR, MANUAL_REVIEW_DIR, METADATA_FILE.parent):
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _load_json(path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SafetyStop(f"JSON okunamadı: {path}: {exc}") from exc

    def _load_checkpoint(self) -> dict[str, Any]:
        if not self.resume or not self.checkpoint_file.exists():
            return {"start": 0, "processed_urls": [], "rejected_records": {}, "retry_counts": {}}
        data = self._load_json(self.checkpoint_file, default={})
        if not isinstance(data, dict) or not isinstance(data.get("start", 0), int) or not isinstance(data.get("processed_urls", []), list):
            raise SafetyStop("Checkpoint şeması bozuk; güvenli şekilde devam edilemiyor.")
        return data

    def _save_state(self, start: int) -> None:
        self.report.total_duration_seconds = round(time.monotonic() - self.started_at, 2)
        self.report.rejection_reasons = dict(self.rejection_counts)
        self.report.checkpoint_offset = start
        self.report.average_record_seconds = round(
            self.report.total_duration_seconds / max(self.report.scanned_records, 1), 3
        )
        remaining = max(self.report.total_records - start, 0)
        self.report.eta_seconds = round(remaining * self.report.average_record_seconds, 1) if self.report.total_records else None
        self.report.total_accepted_kt = len(list(KT_PDF_DIR.glob("*.pdf")))
        try:
            atomic_json_write(METADATA_FILE, self.metadata)
            atomic_json_write(self.checkpoint_file, {
                "start": start,
                "processed_urls": sorted(self.processed_urls),
                "accepted_sha256": sorted(self.known_hashes),
                "rejected_records": self.rejected_records,
                "retry_counts": self.retry_counts,
                "validator_version": VALIDATOR_VERSION,
                "collector_version": COLLECTOR_VERSION,
                "started_at": self.checkpoint.get("started_at", datetime.now(timezone.utc).isoformat()),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })
            atomic_json_write(self.report_file, asdict(self.report))
        except OSError as exc:
            raise SafetyStop(f"Metadata/checkpoint yazılamadı: {exc}") from exc

    @staticmethod
    def _looks_blocked(text: str) -> bool:
        value = normalized(text[:10000])
        return any(marker in value for marker in CAPTCHA_MARKERS)

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        try:
            response = self.session.request(method, url, timeout=(10, 30), verify=True, **kwargs)
        except requests.RequestException as exc:
            self._http_failure(str(exc))
            raise
        if response.status_code != 200:
            self._http_failure(f"HTTP {response.status_code}: {url}")
            raise requests.HTTPError(f"HTTP {response.status_code}", response=response)
        self.consecutive_http_errors = 0
        if "text/html" in response.headers.get("Content-Type", "").lower() and self._looks_blocked(response.text):
            raise SafetyStop("CAPTCHA veya giriş ekranı algılandı.")
        return response

    def _http_failure(self, reason: str) -> None:
        self.report.http_errors += 1
        self.consecutive_http_errors += 1
        logger.warning("http_error consecutive=%d reason=%s", self.consecutive_http_errors, reason)
        if self.consecutive_http_errors >= 10:
            raise SafetyStop("Art arda 10 HTTP hatası oluştu.")

    def open_session(self) -> None:
        response = self._request("GET", KUBKT_URL, headers=self.headers)
        soup = BeautifulSoup(response.text, "html.parser")
        token_input = soup.select_one('input[name="_token"]')
        if token_input and token_input.get("value"):
            self.csrf_token = str(token_input["value"])
        if not self.csrf_token:
            match = re.search(r"_token\s*:\s*['\"]([^'\"]+)['\"]", response.text)
            self.csrf_token = match.group(1) if match else None
        if not self.csrf_token:
            raise SafetyStop("TİTCK sayfasında CSRF token bulunamadı; şema değişmiş olabilir.")

    def fetch_page(self, start: int, length: int = 100) -> tuple[list[dict[str, Any]], int]:
        if not self.csrf_token:
            raise SafetyStop("CSRF oturumu başlatılmadı.")
        data = {"draw": str(start // length + 1), "start": str(start), "length": str(length),
                "search[value]": "", "search[regex]": "false", "order[0][column]": "0",
                "order[0][dir]": "asc", "_token": self.csrf_token}
        headers = dict(self.headers, **{"X-Requested-With": "XMLHttpRequest", "Accept": "application/json"})
        response = self._request("POST", AJAX_URL, data=data, headers=headers)
        try:
            payload = response.json()
        except ValueError as exc:
            raise SafetyStop("DataTables yanıtı JSON değil.") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise SafetyStop("TİTCK DataTables cevap şeması değişti.")
        total = payload.get("recordsFiltered", payload.get("recordsTotal"))
        if not isinstance(total, int):
            raise SafetyStop("DataTables toplam kayıt alanı eksik veya geçersiz.")
        self.report.total_records = total
        return payload["data"], total

    def _extract_document_url(self, record: dict[str, Any]) -> str | None:
        return self._extract_url_for_type(record, self.document_type)

    def _extract_url_for_type(self, record: dict[str, Any], document_type: str) -> str | None:
        field_name = "documentPathKt" if document_type == "KT" else "documentPathKub"
        html = record.get(field_name)
        if not isinstance(html, str):
            return None
        match = re.search(r"href=['\"]([^'\"]+)['\"]", html, flags=re.I)
        if not match:
            return None
        url = urljoin(BASE_URL, match.group(1))
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_PDF_HOSTS:
            raise SafetyStop(f"Resmî olmayan PDF alan adı algılandı: {url}")
        return url

    @staticmethod
    def _approval_date(record: dict[str, Any]) -> str | None:
        for key in (
            "confirmationDateKt", "confirmationDateKub", "approvalDate",
            "approval_date", "publishDate", "date", "updatedAt",
        ):
            value = record.get(key)
            if value not in (None, ""):
                return str(value)
        return None

    def _failure(self, product: str, url: str, reason: str) -> None:
        self.rejection_counts[reason] += 1
        self.report.failed_records.append({"drug_name": product, "source_url": url, "reason": reason})

    def _selection_match(self, record: dict[str, Any]) -> TargetMatch:
        if self.selection_mode == "all":
            return TargetMatch(("all",), ("selection_mode:all",), (), 0, False)
        product, ingredients, atc = TargetSelector.record_fields(record)
        product_norm, ingredients_norm, atc_norm = map(target_normalize, (product, ingredients, atc))
        if self.selection_mode == "targeted":
            match = self.target_selector.match(record) if self.target_selector else TargetMatch()
            available = tuple(
                category for category in match.categories
                if not self.report.category_stats[category]["quota"]
                or self.report.category_stats[category]["accepted"] < self.report.category_stats[category]["quota"]
            )
            if not available:
                return TargetMatch()
            return TargetMatch(available, match.reasons, match.active_ingredients, match.priority, match.combination_product)
        if self.selection_mode == "active-ingredient":
            matched = tuple(item for item in self.requested_active_ingredients if item in ingredients_norm or (not ingredients_norm and item in product_norm))
            return TargetMatch(("active-ingredient",), tuple(f"active_ingredient:{item}" for item in matched), matched, 1, len(matched) > 1) if matched else TargetMatch()
        if self.selection_mode == "atc":
            compact_atc = atc_norm.replace(" ", "")
            matched = tuple(item for item in self.requested_atc_prefixes if compact_atc.startswith(item))
            return TargetMatch(("atc",), tuple(f"atc_prefix:{item.upper()}" for item in matched), (), 1, False) if matched else TargetMatch()
        if self.selection_mode == "product-list":
            matched = tuple(item for item in self.requested_products if item == product_norm or item in product_norm)
            return TargetMatch(("product-list",), ("product_list",), (), 1, False) if matched else TargetMatch()
        return TargetMatch()

    def _category_event(self, match: TargetMatch, event: str, product: str = "") -> None:
        for category in match.categories:
            stats = self.report.category_stats.get(category)
            if not stats:
                continue
            if event in stats and isinstance(stats[event], int):
                stats[event] += 1
            if product and event == "accepted":
                stats["unique_products"] = sorted(set(stats["unique_products"]) | {product})
            if match.active_ingredients and event == "accepted":
                stats["unique_ingredients"] = sorted(set(stats["unique_ingredients"]) | set(match.active_ingredients))

    @staticmethod
    def _compact(value: str) -> str:
        """OCR/layout spaces and punctuation independent comparison form."""
        return re.sub(r"[^a-z0-9]", "", normalized(value))

    @staticmethod
    def _product_tokens(product_name: str) -> list[str]:
        stop = {
            "mg", "ml", "iv", "tablet", "film", "surup", "cozelti", "cozeltisi",
            "solusyon", "solusyonu", "infuzyonluk", "enjeksiyonluk", "ampul", "flakon",
            "oral", "pediatrik", "sudaki", "iceren", "damar", "uygulanir", "hipertonik",
        }
        tokens = re.findall(r"[a-z]+", normalized(product_name))
        return [token for token in tokens if len(token) >= 4 and token not in stop]

    def _verify_pdf(self, content: bytes, product_name: str) -> ValidationResult:
        if not content.startswith(b"%PDF"):
            return ValidationResult("rejected", "PDF imzası yok", 0, 0, [], 0, False, None)
        try:
            reader = pypdf.PdfReader(io.BytesIO(content))
            page_count = len(reader.pages)
            text = " ".join((page.extract_text() or "") for page in reader.pages)
        except Exception as exc:
            return ValidationResult("rejected", f"PDF okunamadı: {type(exc).__name__}", 0, 0, [], 2, False, None)
        if not page_count or len(text.strip()) < 100:
            return ValidationResult("manual_review", "Metin katmanı yok; kontrollü OCR gerekli", page_count,
                                    len(text), [], 4, False, None)
        text_norm = normalized(text)
        compact = self._compact(text)
        kt_phrases = {
            "kullanma_talimati": "kullanma talimati",
            "bu_ilac_nedir": "bu ilac nedir ve ne icin kullanilir",
            "kullanmadan_once": "bu ilaci kullanmadan once dikkat edilmesi gerekenler",
            "nasil_kullanilir": "nasil kullanilir",
            "olasi_yan_etkiler": "olasi yan etkiler nelerdir",
            "saklama": "ilacin saklanmasi",
            "saklama_bilgisi": "saklanmasina iliskin bilgiler",
        }
        kub_phrases = {
            "kisa_urun_bilgisi": "kisa urun bilgisi", "klinik_ozellikler": "klinik ozellikler",
            "farmakolojik_ozellikler": "farmakolojik ozellikler",
        }
        kt_found = [name for name, phrase in kt_phrases.items() if self._compact(phrase) in compact]
        kub_found = [name for name, phrase in kub_phrases.items() if self._compact(phrase) in compact]
        detected_type = "KT" if len(kt_found) >= 2 or "kullanma_talimati" in kt_found else ("KUB" if kub_found else None)
        product_tokens = self._product_tokens(product_name)
        matched_tokens = [token for token in product_tokens if self._compact(token) in compact]
        product_match = bool(matched_tokens)
        requested_headings = kt_found if self.document_type == "KT" else kub_found
        intro = "kullanma_talimati" in kt_found if self.document_type == "KT" else "kisa_urun_bilgisi" in kub_found
        # Legal references such as "yönetmelik" commonly occur inside genuine KT
        # documents. Treat them as negative evidence only when the official
        # document introduction is absent and the term occurs near the beginning.
        opening = compact[:4000]
        unrelated = [marker for marker in UNRELATED_MARKERS if self._compact(marker) in opening and not intro]
        score = 2 + 2 + 2  # official URL is checked before download, signature, readable text
        score += 2 if product_match else 0
        score += 2 if intro else 0
        score += 3 if len(requested_headings) >= 2 else 0
        score += 3 if detected_type == self.document_type else 0
        score -= 5 if unrelated else 0
        score -= 4 if not product_tokens and not product_match else 0
        if detected_type and detected_type != self.document_type:
            return ValidationResult("rejected", f"Belge türü {detected_type}; beklenen {self.document_type}",
                                    page_count, len(text), requested_headings + kub_found, score, product_match, detected_type)
        if unrelated:
            return ValidationResult("rejected", f"İlgisiz kurumsal belge: {', '.join(unrelated)}",
                                    page_count, len(text), requested_headings, score, product_match, detected_type)
        if score >= 10 and intro:
            return ValidationResult("accepted", "Güvenli validator skorunu geçti", page_count,
                                    len(text), requested_headings, score, product_match, detected_type)
        reason = "Ürün adı eşleşmedi" if not product_match else f"{self.document_type} kanıt skoru yetersiz"
        return ValidationResult("rejected", reason, page_count, len(text), requested_headings,
                                score, product_match, detected_type)

    def _check_disk(self, expected_bytes: int = 50 * 1024 * 1024) -> None:
        if shutil.disk_usage(self.accepted_dir).free < expected_bytes:
            raise SafetyStop("Disk alanı yetersiz.")

    def process_record(self, record: dict[str, Any]) -> None:
        product = str(record.get("name") or record.get("productName") or "").strip()
        if not product:
            self.report.rejected += 1
            self._failure("", "", "Ürün adı eksik")
            return
        match = self._selection_match(record)
        if not match.selected:
            return
        self._category_event(match, "candidates", product)
        canonical = canonical_product_fields(product, match.active_ingredients)
        group_key = "|".join([
            target_normalize("+".join(match.active_ingredients)) or canonical["normalized_brand"],
            target_normalize(canonical["strength"]), target_normalize(canonical["dosage_form"]),
            target_normalize(canonical["release_type"]), str(match.combination_product),
        ])
        if self.selection_mode != "all" and self.group_counts[group_key] >= self.max_products_per_ingredient_form:
            return
        url = self._extract_document_url(record)
        if not url:
            return
        self.report.kt_links_found += int(self.document_type == "KT")
        if url in self.processed_urls and not (self.retry_rejected and url in self.rejected_records):
            return
        if self.retry_rejected and url in self.rejected_records:
            self.processed_urls.discard(url)
            self.retry_counts[url] = self.retry_counts.get(url, 0) + 1
        self._check_disk()
        self.report.download_attempts += 1
        self._category_event(match, "downloaded", product)
        response = self._request("GET", url, headers=self.headers)
        content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type in {"text/html", "application/xhtml+xml"}:
            raise SafetyStop("PDF yerine HTML dönmeye başladı.")
        self.processed_urls.add(url)
        if content_type != "application/pdf":
            self.report.rejected += 1
            self._failure(product, url, f"Geçersiz Content-Type: {content_type or 'eksik'}")
            self.rejected_records[url] = f"Geçersiz Content-Type: {content_type or 'eksik'}"
            self.recent_decisions.append(("rejected", "content_type"))
            return
        content = response.content
        self.report.downloaded_pdf += 1
        digest = hashlib.sha256(content).hexdigest()
        if digest in self.known_hashes:
            self.report.duplicates += 1
            self._category_event(match, "duplicate", product)
            return
        quarantine = QUARANTINE_DIR / f"{self.document_type.lower()}_{digest[:12]}.pdf"
        quarantine.write_bytes(content)
        result = self._verify_pdf(content, product)
        if result.decision == "manual_review":
            quarantine.replace(MANUAL_REVIEW_DIR / quarantine.name)
            self.report.manual_review += 1
            self._category_event(match, "manual_review", product)
            self.report.ocr_required += 1
            self._failure(product, url, result.reason)
            self.rejected_records[url] = result.reason
            self.recent_decisions.append(("manual_review", "ocr_required"))
            return
        if result.decision != "accepted":
            quarantine.replace(REJECTED_PDF_DIR / quarantine.name)
            self.report.rejected += 1
            self._category_event(match, "rejected", product)
            self.report.validation_errors += 1
            self.report.product_match_errors += int(not result.product_match)
            self._failure(product, url, result.reason)
            self.rejected_records[url] = result.reason
            self.recent_decisions.append(("rejected", result.reason))
            logger.info("document_rejected product=%s reason=%s score=%d url=%s", product, result.reason, result.score, url)
            return
        safe_product = re.sub(r"[^a-z0-9]+", "_", normalized(product)).strip("_")[:60] or "medicine"
        filename = f"{safe_product}_{self.document_type.lower()}_{digest[:8]}.pdf"
        quarantine.replace(self.accepted_dir / filename)
        legacy_rejected = REJECTED_PDF_DIR / f"{self.document_type.lower()}_{digest[:12]}.pdf"
        if legacy_rejected.exists():
            legacy_rejected.unlink()
        product, record_ingredients, atc_code = TargetSelector.record_fields(record)
        canonical = canonical_product_fields(product, match.active_ingredients)
        self.metadata[filename] = {
            "file_name": filename, "drug_name": product, "document_type": self.document_type,
            "product_name": product, "normalized_product_name": target_normalize(product),
            **canonical,
            "atc_code": atc_code,
            "collection_categories": list(match.categories),
            "selection_reason": list(match.reasons),
            "combination_product": match.combination_product,
            "page_count": result.page_count, "pages": result.page_count, "approval_date": self._approval_date(record),
            "source_url": url, "sha256": digest,
            "download_time": datetime.now(timezone.utc).isoformat(), "status": "success",
            "validation_score": result.score, "detected_headings": result.detected_headings,
            "validator_version": VALIDATOR_VERSION, "validation_status": "accepted",
            "kub_source_url": self._extract_url_for_type(record, "KUB") if self.document_type == "KT" else None,
        }
        self.rejected_records.pop(url, None)
        self.known_hashes.add(digest)
        self.report.accepted += 1
        self.group_counts[group_key] += 1
        self._category_event(match, "accepted", product)
        self.recent_decisions.append(("accepted", "accepted"))
        logger.info("document_accepted product=%s pages=%d score=%d file=%s", product, result.page_count, result.score, filename)

    def _check_acceptance_rate(self) -> None:
        if len(self.recent_decisions) < 20:
            return
        accepted = sum(decision == "accepted" for decision, _ in self.recent_decisions)
        rate = accepted / len(self.recent_decisions)
        if rate < 0.70:
            logger.warning("rolling_acceptance_below_70 window=%d rate=%.3f", len(self.recent_decisions), rate)
        if len(self.recent_decisions) >= 50 and rate < 0.30:
            reasons = Counter(reason for decision, reason in self.recent_decisions if decision != "accepted")
            dominant_reason, dominant_count = reasons.most_common(1)[0]
            if dominant_count > len(self.recent_decisions) / 2:
                raise SafetyStop(
                    f"Son {len(self.recent_decisions)} kayıtta kabul %{rate * 100:.1f}; "
                    f"baskın ret nedeni: {dominant_reason}. Validator/akış incelenmeli."
                )

    def validate_rag(self, sample_size: int = 10) -> None:
        from src.query_analysis import extract_product_name
        from src.rag_chain import MedicalRAGChain, initialize_database_if_empty
        from src.vector_store import vectorstore_status

        rebuild = initialize_database_if_empty()
        status = vectorstore_status()
        self.report.faiss_status = status["status"]
        self.report.total_chunks = status["total_chunks"]
        self.report.total_pdfs = status["pdf_count"]
        self.report.unique_medicines = status["unique_medicines"]
        self.report.manifest_valid = status["manifest_valid"]
        if rebuild.get("total_chunks", 0) < 1 or not status["manifest_valid"]:
            raise SafetyStop("FAISS yeniden oluşturma/manifest doğrulaması başarısız.")
        products = sorted({str(v.get("drug_name", "")) for v in self.metadata.values()
                           if isinstance(v, dict) and v.get("status") == "success" and v.get("drug_name")})
        selected = random.Random(20260721).sample(products, min(sample_size, len(products)))
        chain = MedicalRAGChain(api_key=None, llm=None)
        query_templates = ("{name} nasıl kullanılır?", "{name} yan etkileri nelerdir?", "{name} nedir?")
        failures = 0
        for product in selected:
            alias = extract_product_name(product) or normalized(product).split()[0]
            for template in query_templates:
                query = template.format(name=alias)
                result = chain.ask(query)
                sources = result.get("sources", [])
                source_ok = bool(sources) and all(
                    str(s.get("source", "")).lower().endswith(".pdf")
                    and s.get("page") not in (None, "") and s.get("score") is not None
                    and bool(s.get("drug_name")) for s in sources
                )
                passed = result.get("answer_mode") == "medicine_rag" and source_ok
                failures += int(not passed)
                self.report.rag_tests.append({"drug_name": product, "query": query, "passed": passed,
                                              "source_count": len(sources), "sources": sources})
        if failures:
            raise SafetyStop(f"RAG pilot doğrulamasında {failures} sorgu başarısız oldu.")

    def revalidate_existing(self) -> list[dict[str, Any]]:
        """Audit legacy rejected PDFs without inventing missing source metadata."""
        results: list[dict[str, Any]] = []
        for path in sorted(REJECTED_PDF_DIR.glob(f"{self.document_type.lower()}_*.pdf")):
            content = path.read_bytes()
            # Legacy rejected files did not retain product/URL sidecars. An empty
            # product deliberately prevents automatic acceptance without provenance.
            result = self._verify_pdf(content, "")
            classification = (
                "boş veya taranmış PDF" if result.decision == "manual_review"
                else "gerçek KT ama eski doğrulama kuralı fazla katı" if result.detected_type == "KT"
                else "gerçek KÜB ama KT olarak işlenmiş" if result.detected_type == "KUB"
                else "bilinmeyen"
            )
            entry = {"file_name": path.name, "document_type": self.document_type,
                     "http_status": None, "content_type": "application/pdf",
                     "pdf_signature": content.startswith(b"%PDF"), "classification": classification,
                     **asdict(result)}
            results.append(entry)
            if result.decision == "manual_review":
                path.replace(MANUAL_REVIEW_DIR / path.name)
        audit_file = METADATA_FILE.parent / f"collector_{self.document_type.lower()}_revalidation.json"
        atomic_json_write(audit_file, {
            "validator_version": VALIDATOR_VERSION,
            "sample_size": len(results),
            "note": "Eski dosyalarda kaynak URL/ürün sidecar'ı bulunmadığından otomatik kabul yapılmadı.",
            "results": results,
        })
        return results

    def run(self, postprocess: bool = True) -> Report:
        start = self.from_offset if self.from_offset is not None else (int(self.checkpoint.get("start", 0)) if self.resume else 0)
        since_flush = 0
        try:
            self.open_session()
            while self.limit == 0 or self.report.accepted < self.limit:
                records, total = self.fetch_page(start)
                if not records or start >= total:
                    break
                for record in records:
                    self.report.scanned_records += 1
                    attempts_before = self.report.download_attempts
                    try:
                        self.process_record(record)
                    except requests.RequestException as exc:
                        self.report.failed_records.append({"drug_name": str(record.get("name", "")),
                                                           "source_url": "", "reason": str(exc)})
                    self._check_acceptance_rate()
                    since_flush += 1
                    if since_flush >= self.batch_size:
                        self._save_state(start)
                        since_flush = 0
                    if self.limit and self.report.accepted >= self.limit:
                        break
                    if self.report.download_attempts > attempts_before:
                        time.sleep(self.delay)
                start += len(records)
                self._save_state(start)
            if postprocess and self.report.accepted:
                self.validate_rag()
        except SafetyStop as exc:
            self.report.stop_reason = str(exc)
            logger.error("collector_safety_stop reason=%s", exc)
        finally:
            self._save_state(start)
        return self.report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kontrollü TİTCK KT/KÜB belge toplayıcı")
    parser.add_argument("--type", choices=("kt", "kub"), default="kt")
    parser.add_argument("--limit", type=int, default=20, help="0=tüm arşiv")
    parser.add_argument("--delay", type=float, default=1.5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-rejected", action="store_true")
    parser.add_argument("--revalidate", action="store_true")
    parser.add_argument("--from-offset", type=int)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--no-postprocess", action="store_true")
    parser.add_argument(
        "--selection-mode",
        choices=("all", "targeted", "active-ingredient", "atc", "product-list"),
        default="all",
    )
    parser.add_argument("--categories", default="", help="Virgülle ayrılmış hedef kategori adları")
    parser.add_argument("--active-ingredients", default="", help="Virgülle ayrılmış etkin maddeler")
    parser.add_argument("--atc-prefixes", default="", help="Virgülle ayrılmış ATC önekleri")
    parser.add_argument("--product-list", default="", help="Virgülle ayrılmış tam ürün adları")
    parser.add_argument("--max-products-per-ingredient-form", type=int, default=5)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit negatif olamaz")
    if args.batch_size < 1 or args.batch_size > 1000:
        parser.error("--batch-size 1 ile 1000 arasında olmalıdır")
    if args.from_offset is not None and args.from_offset < 0:
        parser.error("--from-offset negatif olamaz")
    if args.max_products_per_ingredient_form < 1:
        parser.error("--max-products-per-ingredient-form en az 1 olmalıdır")
    args.categories = tuple(item.strip() for item in args.categories.split(",") if item.strip())
    args.active_ingredients = tuple(item.strip() for item in args.active_ingredients.split(",") if item.strip())
    args.atc_prefixes = tuple(item.strip() for item in args.atc_prefixes.split(",") if item.strip())
    args.product_list = tuple(item.strip() for item in args.product_list.split(",") if item.strip())
    required = {
        "active-ingredient": args.active_ingredients,
        "atc": args.atc_prefixes,
        "product-list": args.product_list,
    }
    if args.selection_mode in required and not required[args.selection_mode]:
        parser.error(f"--selection-mode {args.selection_mode} için ilgili filtre listesi zorunludur")
    return args


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = parse_args()
    try:
        collector = TITCKCollector(args.type, args.limit, args.delay, args.resume,
                                   args.batch_size, args.retry_rejected, args.from_offset,
                                   args.selection_mode, args.categories, args.active_ingredients,
                                   args.atc_prefixes, args.product_list,
                                   args.max_products_per_ingredient_form)
        if args.revalidate:
            print(json.dumps(collector.revalidate_existing(), ensure_ascii=False, indent=2))
        report = collector.run(not args.no_postprocess)
    except SafetyStop as exc:
        logger.error("collector_start_failed reason=%s", exc)
        return 2
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    return 2 if report.stop_reason else 0


if __name__ == "__main__":
    raise SystemExit(main())
