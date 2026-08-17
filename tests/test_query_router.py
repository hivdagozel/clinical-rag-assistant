import pytest
from fastapi.testclient import TestClient

from app import app
from src.query_router import QueryRoute, QueryRouter
from src.rag_chain import MedicalRAGChain


class NoRetrievalExpected:
    def retrieve(self, query):
        raise AssertionError("Symptom/emergency routes must not search medicine PDFs")


@pytest.mark.offline
@pytest.mark.parametrize("query", ["Başım ağrıyor.", "Baş ağrısı için ne yapmalıyım?", "Karnım ağrıyor.", "Midem bulanıyor.", "Ateşim var."])
def test_symptoms_route_without_medicine(query):
    result = QueryRouter().classify(query)
    assert result.route == QueryRoute.SYMPTOM
    assert result.product is None


@pytest.mark.offline
def test_headache_triage_is_safe_and_does_not_retrieve():
    result = MedicalRAGChain(llm=None, retriever=NoRetrievalExpected()).ask("Başım ağrıyor.")
    assert result["answer_mode"] == "symptom_triage"
    assert result["answer"]
    assert result["sources"] == []
    assert result["needs_follow_up"] is True
    assert "size uygun seçenek" in result["answer"].casefold()
    assert "belge" not in result["answer"].casefold()


@pytest.mark.offline
def test_headache_triage_explains_common_otc_options_with_safety_limits():
    result = MedicalRAGChain(llm=None, retriever=NoRetrievalExpected()).ask(
        "Başım ağrıyor ne gibi ilaçlar önerirsin?"
    )
    answer = result["answer"].casefold()
    assert result["answer_mode"] == "symptom_triage"
    assert "parasetamol" in answer
    assert "buprofen" in answer
    assert "karaciğer" in answer
    assert "mide ülseri" in answer
    assert "eczacı" in answer


@pytest.mark.offline
def test_abdominal_triage_asks_required_questions():
    result = MedicalRAGChain(llm=None, retriever=NoRetrievalExpected()).ask("Karnım ağrıyor.")
    answer = result["answer"].casefold()
    assert result["answer_mode"] == "symptom_triage"
    assert all(word in answer for word in ("nerede", "ne zamandır", "şiddet", "kusma", "ateş", "kan"))
    assert "sertlik" in answer and "gebelik" in answer and "bayıl" in answer


@pytest.mark.offline
def test_emergency_route_is_immediate():
    result = MedicalRAGChain(llm=None, retriever=NoRetrievalExpected()).ask("Göğsüm ağrıyor ve nefes alamıyorum.")
    assert result["answer_mode"] == "emergency_triage"
    assert "112" in result["answer"]
    assert result["needs_follow_up"] is False


@pytest.mark.offline
def test_medicine_routes_are_distinct():
    router = QueryRouter()
    assert router.classify("Parol nasıl kullanılır?").route == QueryRoute.MEDICINE_CLINICAL
    assert router.classify("Parol hangi firmaya aittir?").route == QueryRoute.MEDICINE_METADATA
    assert router.classify("Bugün hava nasıl?").route == QueryRoute.UNSUPPORTED


@pytest.mark.integration
def test_symptom_endpoint_returns_200_without_sources():
    with TestClient(app) as client:
        app.state.rag_chain = MedicalRAGChain(llm=None, retriever=NoRetrievalExpected())
        response = client.post("/api/ask", json={"question": "Başım ağrıyor"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["answer"]
        assert payload["answer_mode"] == "symptom_triage"
        assert payload["sources"] == []
        assert "error" not in payload
