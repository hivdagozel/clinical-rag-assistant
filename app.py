from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from src.config import STATIC_DIR, ensure_runtime_directories, settings
from src.logging_config import configure_logging
from src.medicine_api_client import check_api_health
from src.rag_chain import (
    LLMGenerationError,
    LLMNotConfiguredError,
    MedicalRAGChain,
    initialize_database_if_empty,
    llm_configuration_status,
)
from src.vector_store import embedding_identity, get_embedding_dimension, vectorstore_status

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_runtime_directories()
    app.state.application_status = "starting"
    app.state.vectorstore_error = None
    app.state.rag_chain = MedicalRAGChain()
    try:
        settings.validate()
        await asyncio.to_thread(initialize_database_if_empty)
    except Exception as exc:
        app.state.vectorstore_error = str(exc)
        logger.exception("startup_vectorstore_failed")
    app.state.medicine_api_online = await asyncio.to_thread(check_api_health) if settings.use_medicine_api else False
    vector = await asyncio.to_thread(vectorstore_status)
    app.state.application_status = "ready" if vector["status"] == "ready" else "degraded"
    yield


app = FastAPI(title="Medical RAG Assistant API", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Soru alanı boş olamaz.")
        return value


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    started = time.perf_counter()
    response = None
    try:
        response = await call_next(request)
    finally:
        logger.info("request_completed request_id=%s path=%s status_code=%s duration_ms=%.2f", request_id, request.url.path, getattr(response, "status_code", 500), (time.perf_counter() - started) * 1000)
    response.headers["X-Request-ID"] = request_id
    return response


@app.post("/api/ask")
async def ask_question(payload: QueryRequest, request: Request):
    rag_chain: MedicalRAGChain | None = getattr(request.app.state, "rag_chain", None)
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG sistemi hazır değil.")
    try:
        result = await asyncio.to_thread(rag_chain.ask, payload.question, request.state.request_id)
        stats = result.get("retrieval_stats", {})
        logger.info(
            "ask_completed request_id=%s route=%s product=%s no_product_behavior=%s medicine_api_called=%s faiss_called=%s pdf_count=%s llm_called=%s fallback_reason=%s answer_mode=%s answer=%r",
            request.state.request_id, stats.get("route"), stats.get("product"),
            "route_without_product" if not stats.get("product") else "product_detected",
            stats.get("medicine_api_called"), stats.get("faiss_called"), stats.get("pdf_count"),
            stats.get("llm_called"), stats.get("fallback_reason"), result.get("answer_mode"),
            str(result.get("answer", ""))[:300],
        )
        return result
    except LLMNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    except LLMGenerationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    except RuntimeError as exc:
        logger.exception("ask_runtime_error")
        raise HTTPException(status_code=503, detail="Belge arama sistemi şu anda kullanılamıyor.") from exc
    except Exception as exc:
        logger.exception("ask_unexpected_error")
        raise HTTPException(status_code=500, detail="Soru işlenirken beklenmeyen bir hata oluştu.") from exc


@app.get("/api/status")
async def get_system_status(request: Request):
    llm = llm_configuration_status()
    rag_chain = getattr(request.app.state, "rag_chain", None)
    if rag_chain is not None:
        llm["status"] = rag_chain.runtime_status
    provider, model = embedding_identity()
    try:
        dimension = await asyncio.to_thread(get_embedding_dimension)
        embedding_status = "ready"
    except Exception:
        dimension = 0
        embedding_status = "unavailable"
    vector = await asyncio.to_thread(vectorstore_status)
    if getattr(request.app.state, "vectorstore_error", None):
        vector["status"] = "vectorstore_error"
        vector["error"] = "Vektör indeksi hazırlanamadı. Sunucu loglarını kontrol edin."
    medicine_online = await asyncio.to_thread(check_api_health) if settings.use_medicine_api else False
    return {
        "application": getattr(request.app.state, "application_status", "starting"),
        "llm": llm,
        "embeddings": {"status": embedding_status, "provider": provider, "model": model, "dimension": dimension},
        "vectorstore": vector,
        "medicine_api": {"enabled": settings.use_medicine_api, "status": "online" if medicine_online else ("offline" if settings.use_medicine_api else "disabled")},
    }


@app.get("/")
async def get_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=503, detail="Web arayüzü bulunamadı.")
    return FileResponse(index_file)


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=settings.app_host, port=settings.app_port, reload=False)
