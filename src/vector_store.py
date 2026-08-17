"""Embedding selection, FAISS persistence and reproducible index manifests."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import (
    MANIFEST_FILE,
    METADATA_FILE,
    VECTORSTORE_DIR,
    indexed_pdf_directories,
    settings,
)

logger = logging.getLogger(__name__)
SPLITTER_VERSION = "1"
METADATA_SCHEMA_VERSION = "1"
FAKE_EMBEDDING_DIMENSION = 128

_embeddings: Embeddings | None = None
_vector_store: FAISS | None = None


class DeterministicFakeEmbeddings(Embeddings):
    """Stable, non-zero hashed token embeddings for offline tests only."""

    def __init__(self, dimension: int = FAKE_EMBEDDING_DIMENSION):
        self.dimension = dimension

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = text.casefold().split() or [text]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


def reset_runtime_state() -> None:
    global _embeddings, _vector_store
    _embeddings = None
    _vector_store = None


def _get_project_root() -> Path:
    from src.config import PROJECT_ROOT
    return PROJECT_ROOT


def _get_persist_directory() -> str:
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    return str(VECTORSTORE_DIR)


def embedding_identity() -> tuple[str, str]:
    if settings.test_mode:
        return "fake", f"deterministic-hash-{FAKE_EMBEDDING_DIMENSION}"
    return settings.embedding_provider, settings.embedding_model


def get_embeddings() -> Embeddings:
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    provider, model = embedding_identity()
    try:
        if provider == "fake":
            _embeddings = DeterministicFakeEmbeddings()
        elif provider == "huggingface":
            from langchain_huggingface import HuggingFaceEmbeddings
            _embeddings = HuggingFaceEmbeddings(
                model_name=model,
                encode_kwargs={"normalize_embeddings": True},
                query_encode_kwargs={"normalize_embeddings": True},
            )
        elif provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY eksik.")
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            _embeddings = GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)
        elif provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY eksik.")
            from langchain_openai import OpenAIEmbeddings
            _embeddings = OpenAIEmbeddings(model=model)
        else:
            raise RuntimeError(f"Desteklenmeyen embedding sağlayıcısı: {provider}")
    except Exception as exc:
        logger.exception("embedding_load_failed provider=%s model=%s", provider, model)
        raise RuntimeError(f"Seçilen embedding modeli yüklenemedi: {provider}/{model}") from exc
    return _embeddings


def get_embedding_dimension(embeddings: Embeddings | None = None) -> int:
    return len((embeddings or get_embeddings()).embed_query("dimension probe"))


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def indexed_pdf_files() -> list[Path]:
    files: list[Path] = []
    for directory in indexed_pdf_directories():
        if directory.exists():
            files.extend(directory.rglob("*.pdf"))
    return sorted({path.resolve() for path in files}, key=lambda p: str(p).casefold())


def calculate_pdf_hashes() -> dict[str, str]:
    from src.config import PROJECT_ROOT
    result: dict[str, str] = {}
    for path in indexed_pdf_files():
        try:
            result[str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")] = _sha256(path)
        except OSError:
            logger.exception("pdf_hash_failed path=%s", path)
    return result


def calculate_pdf_dir_hash() -> str:
    payload = json.dumps(calculate_pdf_hashes(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _metadata_hash() -> str:
    return _sha256(METADATA_FILE) if METADATA_FILE.exists() else hashlib.sha256(b"").hexdigest()


def expected_manifest_config() -> dict[str, Any]:
    provider, model = embedding_identity()
    return {
        "pdf_hashes": calculate_pdf_hashes(),
        "metadata_hash": _metadata_hash(),
        "embedding_provider": provider,
        "embedding_model": model,
        "embedding_dimension": get_embedding_dimension(),
        "embedding_normalized": provider == "huggingface",
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "splitter_version": SPLITTER_VERSION,
        "metadata_schema_version": METADATA_SCHEMA_VERSION,
    }


def read_manifest(path: Path = MANIFEST_FILE) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except (OSError, json.JSONDecodeError):
        logger.exception("manifest_read_failed path=%s", path)
        return None


def manifest_mismatch_reasons(manifest: dict[str, Any] | None = None) -> list[str]:
    manifest = manifest if manifest is not None else read_manifest()
    if not manifest:
        return ["manifest_missing_or_invalid"]
    expected = expected_manifest_config()
    return [key for key, value in expected.items() if manifest.get(key) != value]


def check_and_update_manifest() -> bool:
    reasons = manifest_mismatch_reasons()
    if reasons:
        logger.warning("manifest_mismatch reasons=%s", reasons)
    return not reasons


def _write_manifest(directory: Path, total_documents: int, total_chunks: int, unique_medicines: int) -> dict[str, Any]:
    manifest = {
        **expected_manifest_config(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_documents": total_documents,
        "total_chunks": total_chunks,
        "unique_medicines": unique_medicines,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def save_manifest() -> None:
    count = get_db_stats()
    _write_manifest(VECTORSTORE_DIR, len(indexed_pdf_files()), count, get_unique_medicine_count())


def _load_store(directory: Path) -> FAISS:
    index_path = directory / "index.faiss"
    pickle_path = directory / "index.pkl"
    if not index_path.exists() or not pickle_path.exists():
        raise FileNotFoundError(f"FAISS indeks dosyaları bulunamadı: {directory}")
    store = FAISS.load_local(str(directory), get_embeddings(), allow_dangerous_deserialization=True)
    dimension = int(store.index.d)
    expected_dimension = get_embedding_dimension()
    if dimension != expected_dimension:
        raise RuntimeError(f"Embedding boyutu uyumsuz: indeks={dimension}, model={expected_dimension}")
    return store


def get_vector_store(strict: bool = False) -> FAISS | None:
    global _vector_store
    if _vector_store is not None:
        return _vector_store
    try:
        _vector_store = _load_store(VECTORSTORE_DIR)
        return _vector_store
    except FileNotFoundError:
        logger.warning("vectorstore_not_found path=%s", VECTORSTORE_DIR)
    except Exception:
        logger.exception("vectorstore_load_failed path=%s", VECTORSTORE_DIR)
    if strict:
        raise RuntimeError("FAISS vektör indeksi kullanılamıyor.")
    return None


def add_documents_to_store(documents: List[Document]) -> FAISS:
    """Build and atomically activate a complete index from documents."""
    global _vector_store
    if not documents:
        raise ValueError("Vektör indeksine eklenecek belge yok.")
    VECTORSTORE_DIR.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="vectorstore-build-", dir=str(VECTORSTORE_DIR.parent)))
    backup_dir = VECTORSTORE_DIR.with_name(VECTORSTORE_DIR.name + ".backup")
    logger.info("vectorstore_rebuild_started chunks=%d temp=%s", len(documents), temp_dir)
    try:
        new_store = FAISS.from_documents(documents, get_embeddings())
        new_store.save_local(str(temp_dir))
        unique = {str(doc.metadata.get("normalized_drug_name") or doc.metadata.get("drug_name", "")).casefold() for doc in documents}
        unique.discard("")
        _write_manifest(temp_dir, len(indexed_pdf_files()), len(documents), len(unique))
        verified = _load_store(temp_dir)
        if int(verified.index.ntotal) != len(documents):
            raise RuntimeError("Yeni FAISS indeksindeki chunk sayısı doğrulanamadı.")
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        if VECTORSTORE_DIR.exists() and (VECTORSTORE_DIR / "index.faiss").exists():
            shutil.copytree(VECTORSTORE_DIR, backup_dir)
        VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
        # Keep the stable directory (and therefore its inherited Windows ACLs).
        # The manifest is replaced last and acts as the commit marker.
        for filename in ("index.faiss", "index.pkl", "manifest.json"):
            os.replace(temp_dir / filename, VECTORSTORE_DIR / filename)
        _vector_store = _load_store(VECTORSTORE_DIR)
        logger.info("vectorstore_rebuild_completed chunks=%d path=%s", len(documents), VECTORSTORE_DIR)
        return _vector_store
    except Exception:
        logger.exception("vectorstore_rebuild_failed temp=%s", temp_dir)
        raise
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def semantic_search(query: str, k: int | None = None, distance_threshold: float | None = None,
                    metadata_filter: Any | None = None) -> List[tuple[Document, float]]:
    store = get_vector_store(strict=True)
    threshold = (4.0 if settings.test_mode else settings.distance_threshold) if distance_threshold is None else distance_threshold
    requested_k = k or settings.retrieval_k
    kwargs: dict[str, Any] = {}
    if metadata_filter is not None:
        kwargs["filter"] = metadata_filter
        kwargs["fetch_k"] = min(int(store.index.ntotal), max(requested_k * 100, 1000))
    results = store.similarity_search_with_score(query, k=requested_k, **kwargs)
    selected: list[tuple[Document, float]] = []
    for document, raw_score in results:
        score = float(raw_score)
        logger.info("faiss_result score=%.6f source=%s", score, document.metadata.get("source"))
        if score <= threshold:
            selected.append((document, score))
    return selected


def get_db_stats() -> int:
    store = get_vector_store()
    return int(store.index.ntotal) if store is not None else 0


def get_unique_medicine_count() -> int:
    store = get_vector_store()
    if store is None:
        return 0
    values = {
        str(doc.metadata.get("normalized_drug_name") or doc.metadata.get("drug_name", "")).casefold()
        for doc in store.docstore._dict.values()
    }
    values.discard("")
    return len(values)


def vectorstore_status() -> dict[str, Any]:
    manifest = read_manifest()
    reasons: list[str] = []
    try:
        reasons = manifest_mismatch_reasons(manifest)
    except Exception as exc:
        reasons = [f"config_check_failed:{type(exc).__name__}"]
    count = get_db_stats()
    return {
        "status": "ready" if count > 0 and not reasons else ("missing" if count == 0 else "manifest_mismatch"),
        "path": str(VECTORSTORE_DIR),
        "total_chunks": count,
        "unique_medicines": get_unique_medicine_count(),
        "pdf_count": len(indexed_pdf_files()),
        "manifest_valid": not reasons,
        "manifest_mismatch_reasons": reasons,
    }
