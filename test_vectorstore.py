"""
=============================================================
Test: Aşama 3 - Vektör Veritabanı ve Embedding Testi
=============================================================
Bu script:
  1. ChromaDB ve HuggingFace embedding modelini yükler
  2. Sahte tıbbi metinler oluşturup böler
  3. Böldüğü metinleri Vektör Veritabanına (ChromaDB) yazar
  4. Anlamsal arama (Semantic Search) yaparak sistemi test eder
=============================================================
"""

import sys
from pathlib import Path

# Proje kökünü path'e ekle
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from langchain_core.documents import Document
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

console = Console()

def create_dummy_medical_docs():
    """Test için sahte tıbbi dökümanlar oluşturur."""
    docs = [
        Document(
            page_content="Parol 500mg, hafif ve orta şiddetli ağrılar ile ateş tedavisinde kullanılan etkili bir parasetamol ilacıdır.",
            metadata={"source": "parol_prospektus.pdf", "type": "pdf"}
        ),
        Document(
            page_content="Augmentin 1000mg, bakteriyel enfeksiyonların (örneğin solunum yolu enfeksiyonları) tedavisinde kullanılan geniş spektrumlu bir antibiyotiktir.",
            metadata={"source": "augmentin_prospektus.pdf", "type": "pdf"}
        ),
        Document(
            page_content="Lansor 30mg, mide asidini azaltarak mide ülseri ve reflü tedavisinde kullanılan bir proton pompası inhibitörüdür.",
            metadata={"source": "lansor_prospektus.pdf", "type": "pdf"}
        ),
        Document(
            page_content="Aspirin 100mg, kalp krizi riskini azaltmak ve kanı sulandırmak amacıyla düşük dozlarda günlük olarak kullanılabilir.",
            metadata={"source": "aspirin_prospektus.pdf", "type": "pdf"}
        )
    ]
    return docs

def main():
    console.print(Panel.fit(
        "[bold cyan]Medical RAG Assistant[/bold cyan]\n"
        "[dim]Asama 3: Embedding ve Vektor DB Testi[/dim]",
        border_style="cyan"
    ))

    # Import işlemini try-except içine alalım ki henüz kurulmamışsa uyarı versin
    try:
        from src.vector_store import add_documents_to_store, semantic_search, get_db_stats
        from src.text_splitter import split_documents
    except ImportError as e:
        console.print(f"[red]Gerekli modüller yuklenemedi: {e}[/red]")
        console.print("[yellow]Ipucu: Lütfen pip install isleminin bitmesini bekleyin.[/yellow]")
        return

    # 1. Belgeleri Hazırla ve Böl
    console.print(Rule("[bold]1. Belgeleri Hazirlama[/bold]"))
    raw_docs = create_dummy_medical_docs()
    console.print(f"[dim]{len(raw_docs)} sahte belge olusturuldu.[/dim]")
    
    chunks = split_documents(raw_docs, chunk_size=200, chunk_overlap=20)
    
    # 2. Vektör Veritabanına Ekle
    console.print(Rule("[bold]2. Vektor Veritabanina Ekleme (Embedding)[/bold]"))
    add_documents_to_store(chunks)
    get_db_stats()
    
    # 3. Anlamsal Arama Testi
    console.print(Rule("[bold]3. Anlamsal Arama (Semantic Search) Testleri[/bold]"))
    
    test_queries = [
        "Ateşim çıktı ve başım ağrıyor, ne içebilirim?", # Parol bekliyoruz (ağrı/ateş kelimelerinden yakalamalı)
        "Mide yanmasına iyi gelen ilaç hangisi?",      # Lansor bekliyoruz (mide/reflü kelimesinden)
        "Bakteri enfeksiyonu için antibiyotik var mı?" # Augmentin bekliyoruz
    ]
    
    for query in test_queries:
        results = semantic_search(query, k=1)
        
        console.print(f"\n[cyan]Soru:[/cyan] {query}")
        if results:
            doc = results[0]
            console.print(f"[green]En Iyi Eslesen:[/green] {doc.metadata['source']}")
            console.print(f"[dim]Icerik: {doc.page_content}[/dim]")
        else:
            console.print("[red]Sonuc bulunamadi.[/red]")

if __name__ == "__main__":
    main()
