"""Document-grounded medical RAG chain with explicit failure modes."""
from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.config import settings
from src.document_loader import load_documents_from_directory
from src.hybrid_retriever import HybridRetriever
from src.query_analysis import QueryIntent, normalize_text
from src.query_router import QueryRoute, QueryRouter
from src.text_splitter import split_documents
from src.triage import EMERGENCY_RESPONSE, symptom_triage_response
from src.vector_store import add_documents_to_store, check_and_update_manifest, get_db_stats

logger = logging.getLogger(__name__)

NO_DOCUMENT_MESSAGE = "Bu konuda yüklenen resmi KT/KÜB belgelerinde bilgi bulunamadı."
SYMPTOM_GUIDANCE_MESSAGE = (
    "Belirtilerinize göre kişisel ilaç seçimi veya reçete önerisi yapamam. "
    "Baş ağrısının nedeni ve sizin için güvenli seçenek; yaşınız, hastalıklarınız, "
    "kullandığınız diğer ilaçlar, gebelik durumu ve alerjiler gibi bilgilere bağlıdır. "
    "Uygun ürün ve doz için eczacınıza veya doktorunuza danışın. Ani ve çok şiddetli "
    "baş ağrısı, güçsüzlük, konuşma bozukluğu, bilinç değişikliği, ense sertliği ya da "
    "yüksek ateş varsa acil yardım alın."
)
LLM_FAILURE_MESSAGE = "İlgili belge bulundu ancak cevap oluşturulurken teknik bir hata oluştu."
MODEL_NOT_CONFIGURED_MESSAGE = "Yapay zekâ modeli yapılandırılmamış. GEMINI_API_KEY değerini kontrol edin."

SYSTEM_PROMPT = """Yalnızca verilen resmi belge bağlamını kullanarak Türkçe cevap ver.
Kurallar:
- Belge dışında bilgi ekleme ve tahmin yapma.
- Kullanıcının sorduğu noktayı ilk cümlede doğrudan yanıtla; belgedeki ilgisiz bölümleri özetleme.
- Ürünün tam adı veya farmasötik formu sorudakinden farklıysa bunu kısaca belirt, fakat sorunun cevabını belgede bulabiliyorsan yine cevapla.
- Doz veya kullanım bilgisi uydurma; farklı ürünleri karıştırma.
- Bağlam açık değilse bunun belgede açık olmadığını söyle.
- Kaynak adı ve sayfa uydurma. Kaynaklar uygulama tarafından ayrıca gösterilecektir.
- Acil durum ihtimali varsa kısa biçimde sağlık profesyoneline başvurma uyarısı ekle.
- Gereksiz uzun ve tekrarlayan genel uyarılar ekleme.

BAĞLAM:
{context}
"""


class LLMNotConfiguredError(RuntimeError):
    pass


class LLMGenerationError(RuntimeError):
    pass


def llm_configuration_status() -> dict[str, str]:
    provider = settings.llm_provider
    model = settings.gemini_model if provider == "gemini" else ""
    if provider != "gemini":
        return {"status": "unavailable", "provider": provider, "model": model}
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return {"status": "configured" if key else "missing_api_key", "provider": provider, "model": model}


class MedicalRAGChain:
    def __init__(self, api_key: str | None = None, llm: Any | None = None, retriever: HybridRetriever | None = None):
        self.retriever = retriever or HybridRetriever()
        self.router = QueryRouter()
        self.llm = llm
        self.llm_retry_after = 0.0
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.runtime_status = "missing_api_key" if self.llm is None and not self.api_key else "configured"
        if self.llm is None and self.api_key and settings.llm_provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.llm = ChatGoogleGenerativeAI(
                model=settings.gemini_model,
                temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.1")),
                google_api_key=self.api_key,
                max_retries=int(os.getenv("GEMINI_MAX_RETRIES", "1")),
                timeout=float(os.getenv("GEMINI_TIMEOUT_SECONDS", "15")),
            )
            self.runtime_status = "configured"

    @property
    def is_configured(self) -> bool:
        return self.llm is not None

    @staticmethod
    def _is_verified_pdf(document: Document) -> bool:
        return (
            document.metadata.get("source_type", "pdf") == "pdf"
            and str(document.metadata.get("document_type", "")).upper() in {"KT", "KÜB", "KUB"}
        )

    @staticmethod
    def _sources(documents: List[Document]) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        seen: set[tuple[str, Any]] = set()
        for document in documents:
            metadata = document.metadata
            source = str(metadata.get("source", ""))
            page = metadata.get("page")
            display_page = int(page) + 1 if isinstance(page, int) else page
            identity = (source, display_page)
            if identity in seen:
                continue
            seen.add(identity)
            sources.append(
                {
                    "drug_name": metadata.get("drug_name") or metadata.get("medicine_name") or "",
                    "active_ingredients": metadata.get("active_ingredients") or [],
                    "canonical_product_key": metadata.get("canonical_product_key") or "",
                    "document_type": str(metadata.get("document_type", "")).upper(),
                    "source": source,
                    "page": display_page,
                    "source_url": metadata.get("source_url") or metadata.get("api_url") or "",
                    "score": metadata.get("retrieval_score"),
                    "type": metadata.get("source_type", "pdf"),
                }
            )
        return sources

    @staticmethod
    def _grounded_fallback_answer(documents: List[Document], intent: QueryIntent, question: str = "") -> str:
        """Build a short extractive answer when the remote LLM is unavailable."""
        question_norm = normalize_text(question)
        keywords = {
            QueryIntent.CLINICAL_USAGE: ("kullan", "doz", "uygul", "alın", "aliniz", "sıkl", "siklik"),
            QueryIntent.CLINICAL_SAFETY: ("uyarı", "uyari", "yan etki", "kullanmay", "dikkat", "sakla"),
            QueryIntent.GENERAL_DOCUMENT: ("nedir", "içer", "icer", "kullan"),
            QueryIntent.PRODUCT_METADATA: ("firma", "ruhsat", "üretici", "uretici", "barkod"),
        }[intent]
        if "ne icin" in question_norm or "hangi durumda" in question_norm:
            keywords = ("tedavisinde", "tedavisi", "kullanilir", "kullanılır", "endik", "hastalik", "hastalık")

        candidates: list[tuple[int, int, str]] = []
        seen: set[str] = set()
        for document_index, document in enumerate(documents):
            text = re.sub(r"\s+", " ", document.page_content).strip()
            sentences = re.split(r"(?<=[.!?])\s+", text)
            for sentence_index, sentence in enumerate(sentences):
                sentence = sentence.strip()
                normalized = normalize_text(sentence)
                identity = normalized
                if len(sentence) < 25 or identity in seen:
                    continue
                if sentence.endswith("?"):
                    continue
                seen.add(identity)
                keyword_hits = sum(1 for word in keywords if normalize_text(word) in normalized)
                if not keyword_hits:
                    continue
                penalty = 0
                if "bu kullanma talimatinda" in normalized or re.search(r"\.{3,}|\d+\s*$", sentence):
                    penalty += 5
                if "nedir ve ne icin kullanilir" in normalized and len(sentence) < 100:
                    penalty += 4
                candidates.append((keyword_hits * 10 - penalty, -(document_index * 100 + sentence_index), sentence))

        candidates.sort(reverse=True)
        selected = [sentence for score, _, sentence in candidates if score > 0][:3]
        if not selected:
            for document in documents:
                text = re.sub(r"\s+", " ", document.page_content).strip()
                if len(text) >= 25:
                    selected.append(text[:700].rstrip())
                    break
        if not selected:
            return "İlgili resmî belge bulundu ancak okunabilir bir açıklama çıkarılamadı."
        excerpt = "\n\n".join(f"• {sentence}" for sentence in selected)
        return (
            f"Resmî kullanma talimatındaki ilgili bilgi:\n\n{excerpt}\n\n"
            "Kaynak kartında görünen ürün adı ve farmasötik formunu kontrol edin."
        )

    def ask(self, question: str, request_id: str | None = None) -> Dict[str, Any]:
        started = time.perf_counter()
        route = self.router.classify(question)
        base_stats = {
            "route": route.route.value,
            "route_reason": route.reason,
            "product": route.product,
            "intent": route.intent,
            "medicine_api_called": False,
            "faiss_called": False,
            "pdf_count": 0,
            "api_count": 0,
            "llm_called": False,
            "fallback_reason": None,
        }
        logger.info(
            "query_routed request_id=%s route=%s product=%s intent=%s reason=%s",
            request_id, route.route.value, route.product, route.intent, route.reason,
        )
        if route.route == QueryRoute.EMERGENCY:
            base_stats["fallback_reason"] = "emergency_triage"
            return {
                "question": question, "answer": EMERGENCY_RESPONSE, "answer_mode": "emergency_triage",
                "sources": [], "retrieval_stats": base_stats, "needs_follow_up": False,
                "suggested_questions": [],
            }
        if route.route == QueryRoute.SYMPTOM:
            answer, suggestions = symptom_triage_response(question)
            base_stats["fallback_reason"] = "symptom_triage"
            return {
                "question": question, "answer": answer, "answer_mode": "symptom_triage",
                "sources": [], "retrieval_stats": base_stats, "needs_follow_up": True,
                "suggested_questions": suggestions,
            }
        if route.route == QueryRoute.UNSUPPORTED:
            base_stats["fallback_reason"] = "unsupported"
            return {
                "question": question,
                "answer": "Bu asistan yalnızca ilaç belgeleri ve genel belirti yönlendirmesi kapsamındaki soruları yanıtlayabilir.",
                "answer_mode": "unsupported", "sources": [], "retrieval_stats": base_stats,
                "needs_follow_up": False, "suggested_questions": [],
            }

        query_norm = normalize_text(question)
        detail_markers = (
            "tablet", "surup", "suspansiyon", "ampul", "flakon", "kapsul", "sprey",
            "damla", "krem", "merhem", "jel", "iv", "im", "pediatrik", "forte", "hot", "plus",
        )
        has_product_detail = bool(re.search(r"\d", query_norm)) or any(marker in query_norm for marker in detail_markers)
        documents, stats = self.retriever.retrieve(question)
        stats.update({"route": route.route.value, "route_reason": route.reason, "faiss_called": True})
        stats["medicine_api_called"] = bool(settings.use_medicine_api)
        stats.setdefault("llm_called", False)
        intent = QueryIntent(stats["intent"])
        verified_pdfs = [doc for doc in documents if self._is_verified_pdf(doc)]
        product_names = sorted({
            str(doc.metadata.get("drug_name") or doc.metadata.get("medicine_name") or "").strip()
            for doc in verified_pdfs
            if doc.metadata.get("drug_name") or doc.metadata.get("medicine_name")
        })
        if has_product_detail and len(product_names) > 1:
            query_numbers = {value.replace(",", ".") for value in re.findall(r"\d+(?:[.,]\d+)?", question)}
            query_forms = {marker for marker in detail_markers if marker in query_norm}

            def matches_requested_variant(document: Document) -> bool:
                name = str(document.metadata.get("drug_name") or document.metadata.get("medicine_name") or "")
                name_norm = normalize_text(name)
                name_numbers = {value.replace(",", ".") for value in re.findall(r"\d+(?:[.,]\d+)?", name)}
                return query_numbers.issubset(name_numbers) and all(form in name_norm for form in query_forms)

            narrowed = [doc for doc in verified_pdfs if matches_requested_variant(doc)]
            if narrowed:
                verified_pdfs = narrowed
                product_names = sorted({
                    str(doc.metadata.get("drug_name") or doc.metadata.get("medicine_name") or "").strip()
                    for doc in verified_pdfs
                })
        if intent != QueryIntent.PRODUCT_METADATA and not verified_pdfs:
            stats["fallback_reason"] = "medicine_not_found" if not route.product else "document_not_found"
            logger.warning("no_document_found product=%s intent=%s", stats.get("product"), intent.value)
            message = "İlaç adı tespit edilemedi veya bu ilaç sistemde bulunmuyor." if not route.product else NO_DOCUMENT_MESSAGE
            return {"question": question, "answer": message, "answer_mode": "medicine_rag", "sources": [], "retrieval_stats": stats, "needs_follow_up": True, "suggested_questions": ["İlacın tam adını yazar mısınız?"]}
        if intent == QueryIntent.PRODUCT_METADATA and not documents:
            stats["fallback_reason"] = "medicine_api_disabled" if not settings.use_medicine_api else "no_document_found"
            message = "İlaç bilgi servisi kapalı veya erişilemiyor; firma, barkod ve ruhsat bilgisi doğrulanamadı."
            return {"question": question, "answer": message, "answer_mode": "medicine_metadata", "sources": [], "retrieval_stats": stats, "needs_follow_up": False, "suggested_questions": []}
        context_documents = documents if intent == QueryIntent.PRODUCT_METADATA else verified_pdfs
        if not self.is_configured:
            stats["fallback_reason"] = "llm_not_configured"
            answer = self._grounded_fallback_answer(context_documents, intent, question)
            return {"question": question, "answer": answer, "answer_mode": "medicine_rag", "sources": self._sources(context_documents), "retrieval_stats": stats, "needs_follow_up": False, "suggested_questions": []}
        if self.runtime_status == "unavailable" and time.monotonic() < self.llm_retry_after:
            stats["fallback_reason"] = "llm_circuit_open"
            answer = self._grounded_fallback_answer(context_documents, intent, question)
            stats["total_duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
            return {
                "question": question, "answer": answer, "answer_mode": "medicine_rag",
                "sources": self._sources(context_documents), "retrieval_stats": stats,
                "needs_follow_up": False, "suggested_questions": [],
            }

        context = self.retriever.build_context_string(context_documents)
        prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{question}")])
        chain = prompt | self.llm | StrOutputParser()
        llm_started = time.perf_counter()
        stats["llm_called"] = True
        try:
            answer = chain.invoke({"context": context, "question": question})
        except Exception as exc:
            self.runtime_status = "unavailable"
            self.llm_retry_after = time.monotonic() + 60.0
            stats["fallback_reason"] = "llm_generation_failed"
            logger.exception("llm_generation_failed product=%s intent=%s", stats.get("product"), intent.value)
            answer = self._grounded_fallback_answer(context_documents, intent, question)
        stats["llm_duration_ms"] = round((time.perf_counter() - llm_started) * 1000, 2)
        if stats.get("fallback_reason") != "llm_generation_failed":
            self.runtime_status = "ready"
        stats["total_duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return {
            "question": question,
            "answer": str(answer).strip(),
            "answer_mode": "medicine_rag" if route.route == QueryRoute.MEDICINE_CLINICAL else "medicine_metadata",
            "sources": self._sources(context_documents),
            "retrieval_stats": stats,
            "needs_follow_up": False,
            "suggested_questions": [],
        }


def initialize_database_if_empty() -> dict[str, Any]:
    from src.vector_store import manifest_mismatch_reasons
    current_count = get_db_stats()
    reasons = manifest_mismatch_reasons() if current_count else ["index_missing"]
    if not reasons:
        return {"rebuilt": False, "total_chunks": current_count, "reasons": []}
    logger.warning("vectorstore_rebuild_required reasons=%s", reasons)
    raw_documents = load_documents_from_directory()
    if not raw_documents:
        raise RuntimeError("Doğrulanmış KT/KÜB PDF belgesi bulunamadı.")
    chunks = split_documents(raw_documents)
    store = add_documents_to_store(chunks)
    return {"rebuilt": True, "total_chunks": int(store.index.ntotal), "reasons": reasons}
