"""Central, deterministic application configuration and paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _path(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default))
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


DATA_DIR = _path("DATA_DIR", "data")
ACCEPTED_PDF_DIR = _path("ACCEPTED_PDF_DIR", "data/accepted_pdfs")
KT_PDF_DIR = _path("KT_PDF_DIR", "data/accepted_pdfs/kt")
KUB_PDF_DIR = _path("KUB_PDF_DIR", "data/accepted_pdfs/kub")
CLINICAL_GUIDELINE_DIR = _path("CLINICAL_GUIDELINE_DIR", "data/accepted_pdfs/clinical_guidelines")
QUARANTINE_DIR = _path("QUARANTINE_DIR", "data/quarantine")
REJECTED_PDF_DIR = _path("REJECTED_PDF_DIR", "data/rejected_pdfs")
METADATA_FILE = _path("METADATA_FILE", "data/metadata/documents.json")
VECTORSTORE_DIR = _path("TEST_VECTORSTORE_DIR" if os.getenv("TEST_MODE", "false").lower() == "true" else "VECTORSTORE_DIR", "data/vectorstore/medicines")
CLINICAL_VECTORSTORE_DIR = _path("CLINICAL_VECTORSTORE_DIR", "data/vectorstore/clinical_guidelines")
MANIFEST_FILE = VECTORSTORE_DIR / "manifest.json"
STATIC_DIR = _path("STATIC_DIR", "static")


def env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_host: str = os.getenv("APP_HOST", "127.0.0.1")
    app_port: int = int(os.getenv("APP_PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
    test_mode: bool = env_bool("TEST_MODE")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "5"))
    distance_threshold: float = float(os.getenv("FAISS_DISTANCE_THRESHOLD", "1.3"))
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
    )
    llm_provider: str = os.getenv("LLM_PROVIDER", "gemini").lower()
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    use_medicine_api: bool = env_bool("USE_MEDICINE_API", False)
    medicine_api_url: str = os.getenv("MEDICINE_API_URL", "http://localhost:3000").rstrip("/")
    medicine_api_timeout: float = float(os.getenv("MEDICINE_API_TIMEOUT", "10"))

    def validate(self) -> None:
        if self.chunk_size < 100:
            raise ValueError("CHUNK_SIZE en az 100 olmalıdır.")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("CHUNK_OVERLAP, CHUNK_SIZE değerinden küçük ve negatif olmamalıdır.")
        allowed = {"fake", "huggingface", "gemini", "openai"}
        if self.embedding_provider not in allowed:
            raise ValueError(f"Desteklenmeyen EMBEDDING_PROVIDER: {self.embedding_provider}")
        if self.embedding_provider == "fake" and not self.test_mode:
            raise ValueError("fake embedding yalnız TEST_MODE=true iken kullanılabilir.")


settings = Settings()


def indexed_pdf_directories() -> tuple[Path, Path]:
    return KT_PDF_DIR, KUB_PDF_DIR


def ensure_runtime_directories() -> None:
    for path in (KT_PDF_DIR, KUB_PDF_DIR, CLINICAL_GUIDELINE_DIR, VECTORSTORE_DIR, CLINICAL_VECTORSTORE_DIR, QUARANTINE_DIR, REJECTED_PDF_DIR, METADATA_FILE.parent):
        path.mkdir(parents=True, exist_ok=True)
