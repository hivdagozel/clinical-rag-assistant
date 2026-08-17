from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.config import PROJECT_ROOT, KT_PDF_DIR, KUB_PDF_DIR, VECTORSTORE_DIR
from src.hybrid_retriever import HybridRetriever
from src.query_analysis import (
    QueryIntent, classify_intent, extract_product_name, is_symptom_query,
    product_matches, product_matches_metadata,
)
from src.rag_chain import LLMGenerationError, MedicalRAGChain, NO_DOCUMENT_MESSAGE
from src.text_splitter import split_documents
from src.vector_store import (
    DeterministicFakeEmbeddings,
    add_documents_to_store,
    check_and_update_manifest,
    manifest_mismatch_reasons,
    read_manifest,
    reset_runtime_state,
)


def doc(text, drug, source, page=0, doc_type="KT"):
    return Document(page_content=text, metadata={
        "source": source, "page": page, "drug_name": drug,
        "normalized_drug_name": drug.casefold(), "document_type": doc_type,
        "source_type": "pdf",
    })


@pytest.fixture(autouse=True)
def clean_test_index(isolated_vectorstore):
    for child in isolated_vectorstore.iterdir():
        if child.is_dir():
            import shutil; shutil.rmtree(child)
        else:
            child.unlink()
    reset_runtime_state()
    yield
    reset_runtime_state()


@pytest.fixture
def indexed_documents():
    documents = [
        doc("PAROL kullanma talimatı. Uygun doz ve uygulama yolu doktor tarafından belirlenir.", "PAROL 10 MG/ML İNFÜZYON", "parol_kt.pdf", 3),
        doc("PAROL olası yan etkileri arasında bulantı yer alır.", "PAROL 10 MG/ML İNFÜZYON", "parol_kt.pdf", 5),
        doc("PAROL PLUS kullanım bilgileri ve kafein içeriği.", "PAROL PLUS TABLET", "parol_plus_kt.pdf", 2),
        doc("PAROL HOT kullanım bilgileri.", "PAROL HOT SAŞE", "parol_hot_kt.pdf", 2),
        doc("ASPİRİN 25 derecenin altında ve kuru yerde saklanır.", "ASPİRİN 100 MG TABLET", "aspirin_kt.pdf", 8),
    ]
    add_documents_to_store(documents)
    return documents


@pytest.mark.offline
def test_config_paths_are_rooted_and_isolated():
    assert KT_PDF_DIR == PROJECT_ROOT / "data/accepted_pdfs/kt"
    assert KUB_PDF_DIR == PROJECT_ROOT / "data/accepted_pdfs/kub"
    assert VECTORSTORE_DIR != PROJECT_ROOT / "data/vectorstore"


@pytest.mark.offline
def test_fake_embeddings_are_stable_and_nonzero():
    embeddings = DeterministicFakeEmbeddings(64)
    first = embeddings.embed_query("Parol nasıl kullanılır")
    assert first == embeddings.embed_query("Parol nasıl kullanılır")
    assert first != embeddings.embed_query("Aspirin nasıl saklanır")
    assert any(first)


@pytest.mark.offline
def test_manifest_detects_chunk_and_model_changes(indexed_documents, monkeypatch):
    manifest = read_manifest()
    assert manifest and check_and_update_manifest()
    changed = dict(manifest, chunk_size=manifest["chunk_size"] + 1)
    assert "chunk_size" in manifest_mismatch_reasons(changed)
    changed = dict(manifest, embedding_model="another-model")
    assert "embedding_model" in manifest_mismatch_reasons(changed)
    changed = dict(manifest, pdf_hashes={"changed.pdf": "different"})
    assert "pdf_hashes" in manifest_mismatch_reasons(changed)


@pytest.mark.offline
def test_chunking_preserves_metadata_and_has_no_empty_chunks():
    chunks = split_documents([doc("Uzun ve anlamlı bir tıbbi açıklama. " * 30, "PAROL", "parol.pdf")], chunk_size=180, chunk_overlap=30)
    assert chunks
    assert all(chunk.page_content.strip() for chunk in chunks)
    assert all(chunk.metadata["drug_name"] == "PAROL" for chunk in chunks)


@pytest.mark.offline
@pytest.mark.parametrize("query,expected", [("Parol nasıl kullanılır?", "parol"), ("PAROL PLUS dozu", "parol plus"), ("Parol Hot nedir?", "parol hot")])
def test_product_extraction(query, expected):
    assert extract_product_name(query) == expected


@pytest.mark.offline
def test_symptom_recommendation_is_not_treated_as_a_product_name():
    query = "Başım ağrıyor, hangi ilacı kullanmalıyım?"
    assert is_symptom_query(query)
    assert extract_product_name(query) is None


@pytest.mark.offline
def test_product_matching_keeps_variants_separate():
    assert product_matches("parol", "PAROL 500 MG TABLET")
    assert not product_matches("parol", "PAROL PLUS TABLET")
    assert product_matches("parol plus", "PAROL PLUS TABLET")
    assert product_matches("parasetamol", "PAROL 500 MG TABLET")
    assert not product_matches("parasetamol", "PAROL PLUS TABLET")
    assert not product_matches("parol plus", "PAROL HOT SAŞE")


@pytest.mark.offline
def test_intent_classification():
    assert classify_intent("Parol nasıl kullanılır?") == QueryIntent.CLINICAL_USAGE
    assert classify_intent("Parol yan etkileri") == QueryIntent.CLINICAL_SAFETY
    assert classify_intent("Parol hangi firmaya ait?") == QueryIntent.PRODUCT_METADATA


@pytest.mark.offline
def test_keyword_fallback_always_returns_tuples():
    retriever = HybridRetriever([doc("Parol nasıl kullanılır", "PAROL", "parol.pdf")])
    with patch("src.vector_store.get_vector_store", return_value=None):
        results = retriever._search_pdf_documents("Parol nasıl kullanılır")
    assert results and isinstance(results[0], tuple) and len(results[0]) == 2


@pytest.mark.offline
def test_retriever_excludes_plus_and_hot(indexed_documents):
    documents, stats = HybridRetriever().retrieve("Parol nasıl kullanılır?")
    names = [item.metadata["drug_name"] for item in documents]
    assert stats["pdf_count"] > 0
    assert any(name.startswith("PAROL ") for name in names)
    assert not any("PLUS" in name or "HOT" in name for name in names)


@pytest.mark.offline
def test_api_results_do_not_prevent_pdf_search(indexed_documents, monkeypatch):
    from src.config import settings
    previous = settings.use_medicine_api
    object.__setattr__(settings, "use_medicine_api", True)
    api_doc = Document(page_content="Firma: Test", metadata={"source_type": "api", "document_type": "API", "medicine_name": "PAROL", "source": "API"})
    try:
        with patch("src.medicine_api_client.check_api_health", return_value=True), patch("src.medicine_api_client.get_medicine_context", return_value=[api_doc]):
            documents, stats = HybridRetriever().retrieve("Parol hangi firmaya aittir?")
    finally:
        object.__setattr__(settings, "use_medicine_api", previous)
    assert stats["api_count"] == 1
    assert stats["pdf_count"] > 0


@pytest.mark.offline
def test_rag_success_and_programmatic_sources(indexed_documents):
    chain = MedicalRAGChain(llm=FakeListChatModel(responses=["Belgedeki kullanım talimatını izleyiniz."]))
    result = chain.ask("Parol nasıl kullanılır?")
    assert result["answer"]
    assert result["sources"] and result["sources"][0]["page"]
    assert all("PLUS" not in source["drug_name"] for source in result["sources"])


@pytest.mark.offline
def test_rag_no_document_and_llm_failure_are_distinct(indexed_documents):
    chain = MedicalRAGChain(llm=FakeListChatModel(responses=["unused"]))
    missing = chain.ask("Sistemde bulunmayan bir ilaç nasıl kullanılır?")
    assert missing["retrieval_stats"]["fallback_reason"] == "medicine_not_found"
    assert missing["answer"]

    failing = MedicalRAGChain(llm=FakeListChatModel(responses=[]))
    fallback = failing.ask("Parol 120 mg/5 ml oral süspansiyon nasıl kullanılır?")
    assert fallback["answer"]
    assert fallback["sources"]
    assert fallback["retrieval_stats"]["fallback_reason"] == "llm_generation_failed"


@pytest.mark.offline
def test_llm_circuit_breaker_uses_fast_fallback(indexed_documents):
    failing = MedicalRAGChain(llm=FakeListChatModel(responses=[]))
    first = failing.ask("Parol 120 mg/5 ml oral süspansiyon nasıl kullanılır?")
    second = failing.ask("Parol 120 mg/5 ml oral süspansiyon yan etkileri nelerdir?")
    assert first["retrieval_stats"]["fallback_reason"] == "llm_generation_failed"
    assert second["retrieval_stats"]["fallback_reason"] == "llm_circuit_open"
    assert second["answer"]
    assert second["sources"]


@pytest.mark.offline
def test_extractive_fallback_answers_indication_without_repeating_toc():
    documents = [
        Document(
            page_content=(
                "1. ACNEGEN nedir ve ne için kullanılır? "
                "ACNEGEN, şiddetli akne formlarının tedavisinde kullanılır. "
                "Bu Kullanma Talimatında: Olası yan etkiler nelerdir?"
            ),
            metadata={
                "drug_name": "ACNEGEN 10 MG YUMUŞAK JELATİN KAPSÜL",
                "document_type": "KT",
                "source_type": "pdf",
                "source": "acnegen.pdf",
                "page": 0,
            },
        )
    ]
    answer = MedicalRAGChain._grounded_fallback_answer(
        documents, QueryIntent.GENERAL_DOCUMENT, "Acnegen krem ne için kullanılır?"
    )
    assert "şiddetli akne formlarının tedavisinde kullanılır" in answer
    assert answer.count("Bu Kullanma Talimatında") == 0


@pytest.mark.offline
def test_active_ingredient_query_finds_matching_brand(indexed_documents):
    documents, stats = HybridRetriever().retrieve("Parasetamol ilacı nasıl kullanılır?")
    assert stats["pdf_count"] > 0
    assert documents
    assert all("PLUS" not in item.metadata["drug_name"] and "HOT" not in item.metadata["drug_name"] for item in documents)


@pytest.mark.offline
def test_dynamic_product_extraction_preserves_plus_variant(indexed_documents):
    from src.query_analysis import extract_product_name

    product = extract_product_name("ACDKIDS PLUS 2000 IU oral damla nasıl kullanılır?")
    assert product and product.endswith(" plus")


@pytest.mark.offline
def test_dynamic_product_matching_does_not_use_generic_ingredient_tokens():
    """Generic ingredient names must not collapse distinct products."""
    assert product_matches("osel", "% 5 DEKSTROZ %0,45 SODYUM KLORÜR OSEL IV ENJEKSİYONLUK ÇÖZELTİSİ")
    assert not product_matches("osel", "% 0.4 LİDODEKS % 5 DEKSTROZ İÇİNDE İ.V. İNFÜZYON İÇİN ÇÖZELTİ")


@pytest.mark.offline
def test_active_ingredient_metadata_matches_without_brand_collision():
    metadata = {
        "drug_name": "ÖRNEK 500 MG TABLET",
        "active_ingredients": ["parasetamol"],
    }
    assert product_matches_metadata("parasetamol", metadata)
    assert not product_matches_metadata("ibuprofen", metadata)
