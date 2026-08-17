import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app import app
from src.rag_chain import MedicalRAGChain


class StubRetriever:
    def retrieve(self, query):
        document = Document(page_content="PAROL kullanma talimatı.", metadata={
            "source": "parol_kt.pdf", "page": 2, "drug_name": "PAROL",
            "document_type": "KT", "source_type": "pdf", "retrieval_score": 0.2,
        })
        return [document], {"product": "parol", "intent": "clinical_usage", "api_count": 0, "pdf_count": 1, "total": 1}

    def build_context_string(self, documents):
        return "[KAYNAK 1]\nİlaç: PAROL\nBelge türü: KT\nDosya: parol_kt.pdf\nSayfa: 3\nİçerik: PAROL kullanma talimatı."


@pytest.mark.integration
def test_frontend_status_and_ask_endpoints():
    with TestClient(app) as client:
        app.state.rag_chain = MedicalRAGChain(llm=FakeListChatModel(responses=["Belgeye göre kullanınız."]), retriever=StubRetriever())
        assert client.get("/").status_code == 200
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/api/status").status_code == 200
        response = client.post("/api/ask", json={"question": "Parol nasıl kullanılır?"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["answer"]
        assert payload["retrieval_stats"]["pdf_count"] == 1
        assert payload["sources"][0]["document_type"] == "KT"
        assert payload["sources"][0]["page"] == 3
