"""Hybrid retrieval over verified PDF chunks and optional medicine metadata API."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document

from src.config import settings
from src.query_analysis import (
    QueryIntent,
    classify_intent,
    document_priority,
    extract_product_name,
    is_symptom_query,
    normalize_text,
    product_matches,
    product_matches_metadata,
    symptom_terms,
)

logger = logging.getLogger(__name__)
ScoredDocument = tuple[Document, float]


def is_known_medicine(name: str) -> bool:
    return extract_product_name(name) is not None


def _document_product_name(document: Document) -> str:
    metadata = document.metadata
    return str(
        metadata.get("normalized_drug_name")
        or metadata.get("drug_name")
        or metadata.get("medicine_name")
        or Path(str(metadata.get("source", ""))).stem
    )


class HybridRetriever:
    def __init__(self, pdf_documents: List[Document] | None = None):
        self.pdf_documents = pdf_documents or []

    def _keyword_search(self, query: str, top_k: int | None = None) -> list[ScoredDocument]:
        query_words = {word for word in normalize_text(query).split() if len(word) > 2}
        scored: list[ScoredDocument] = []
        for document in self.pdf_documents:
            content_words = set(normalize_text(document.page_content).split())
            overlap = len(query_words.intersection(content_words))
            if overlap:
                scored.append((document, 1.0 / (1.0 + overlap)))
        scored.sort(key=lambda pair: pair[1])
        return scored[: top_k or settings.retrieval_k]

    def _search_pdf_documents(
        self,
        query: str,
        top_k: int | None = None,
        distance_threshold: float | None = None,
        product: str | None = None,
    ) -> list[ScoredDocument]:
        from src.vector_store import get_vector_store, semantic_search

        store = get_vector_store()
        if store is not None:
            candidate_k = min(int(store.index.ntotal), max((top_k or settings.retrieval_k) * 5, 10))
            metadata_filter = None
            if product:
                metadata_filter = lambda metadata: product_matches_metadata(product, metadata)
            return semantic_search(query, k=candidate_k, distance_threshold=distance_threshold,
                                   metadata_filter=metadata_filter)
        return self._keyword_search(query, top_k)

    def retrieve(self, query: str) -> Tuple[List[Document], dict]:
        product = extract_product_name(query)
        symptom_query = is_symptom_query(query)
        query_symptom_terms = symptom_terms(query)
        intent = classify_intent(query)
        stats = {
            "product": product,
            "symptom_query": symptom_query,
            "intent": intent.value,
            "medicine_api_enabled": settings.use_medicine_api,
            "medicine_api_status": "disabled" if not settings.use_medicine_api else "unavailable",
            "api_count": 0,
            "pdf_count": 0,
            "total": 0,
            "faiss_scores": [],
            "fallback_reason": None,
        }
        logger.info("retrieval_started product=%s intent=%s", product, intent.value)

        api_documents: list[Document] = []
        if settings.use_medicine_api:
            from src.medicine_api_client import check_api_health, get_medicine_context
            if check_api_health():
                stats["medicine_api_status"] = "online"
                api_documents = get_medicine_context(query, limit=3)
            for document in api_documents:
                document.metadata.setdefault("source_type", "api")
                document.metadata.setdefault("document_type", "API")
                document.metadata.setdefault("retrieval_score", 0.0)
            if product:
                api_documents = [doc for doc in api_documents if product_matches(product, _document_product_name(doc))]
            stats["api_count"] = len(api_documents)

        expansions = {
            QueryIntent.CLINICAL_USAGE: "3. nasıl kullanılır uygun kullanım doz uygulama sıklığı uygulama yolu",
            QueryIntent.CLINICAL_SAFETY: "uyarılar yan etkiler saklanması gebelik etkileşim",
            QueryIntent.PRODUCT_METADATA: "firma ruhsat sahibi üretici barkod ATC",
            QueryIntent.GENERAL_DOCUMENT: "nedir ne için kullanılır etkin madde",
        }
        retrieval_query = f"{query} {expansions[intent]}"
        pdf_results = self._search_pdf_documents(retrieval_query, product=product) if product or symptom_query else []
        filtered_pdf_results: list[ScoredDocument] = []
        for document, score in pdf_results:
            document.metadata.setdefault("source_type", "pdf")
            document.metadata["document_type"] = str(document.metadata.get("document_type") or document.metadata.get("doc_type") or "").upper()
            if document.metadata["document_type"] not in {"KT", "KÜB", "KUB"}:
                continue
            if product and not product_matches_metadata(product, document.metadata):
                continue
            document.metadata["retrieval_score"] = float(score)
            content = normalize_text(document.page_content)
            if symptom_query and query_symptom_terms and not any(term in content for term in query_symptom_terms):
                continue
            section_boost = 0.0
            if intent == QueryIntent.CLINICAL_USAGE and ("uygun kullanim ve doz" in content or "nasil kullanilir" in content):
                section_boost = 0.45
            elif intent == QueryIntent.CLINICAL_SAFETY and any(section in content for section in ("olasi yan etkiler", "saklanmasi", "kullanmadan once")):
                section_boost = 0.35
            document.metadata["ranking_score"] = float(score) - section_boost
            filtered_pdf_results.append((document, float(score)))

        filtered_pdf_results.sort(key=lambda pair: float(pair[0].metadata.get("ranking_score", pair[1])))
        filtered_pdf_results = filtered_pdf_results[: settings.retrieval_k]
        pdf_documents = [document for document, _ in filtered_pdf_results]
        stats["pdf_count"] = len(pdf_documents)
        stats["faiss_scores"] = [score for _, score in filtered_pdf_results]

        documents = api_documents + pdf_documents
        documents.sort(
            key=lambda document: (
                document_priority(intent, str(document.metadata.get("document_type", "")), str(document.metadata.get("source_type", "pdf"))),
                float(document.metadata.get("retrieval_score", 0.0)),
            )
        )
        stats["total"] = len(documents)
        if not documents:
            stats["fallback_reason"] = "no_document_found"
        logger.info(
            "retrieval_completed product=%s intent=%s api_count=%d pdf_count=%d scores=%s",
            product, intent.value, stats["api_count"], stats["pdf_count"], stats["faiss_scores"],
        )
        return documents, stats

    def build_context_string(self, documents: List[Document]) -> str:
        parts: list[str] = []
        for index, document in enumerate(documents, 1):
            metadata = document.metadata
            page = metadata.get("page")
            display_page = int(page) + 1 if isinstance(page, int) else page or "?"
            parts.extend(
                [
                    f"[KAYNAK {index}]",
                    f"İlaç: {_document_product_name(document)}",
                    f"Belge türü: {metadata.get('document_type', 'Bilinmiyor')}",
                    f"Dosya: {Path(str(metadata.get('source', 'Bilinmiyor'))).name}",
                    f"Sayfa: {display_page}",
                    "İçerik:",
                    document.page_content.strip(),
                    "",
                ]
            )
        return "\n".join(parts) if parts else "İlgili resmi belge bulunamadı."
