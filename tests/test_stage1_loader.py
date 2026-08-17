"""
=============================================================
Aşama 1 Test Scripti - document_loader.py'yi çalıştır ve doğrula
=============================================================

Kullanım:
    python tests/test_stage1_loader.py

Bu script:
  1. data/raw_pdfs klasöründeki PDF'leri yükler
  2. Özet tabloyu gösterir
  3. İlk sayfanın içeriğini inceler
  4. Tüm Document'lerin metadata'sını doğrular
=============================================================
"""

import sys
import os

# Proje kök dizinini Python path'ine ekle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.document_loader import (
    load_documents_from_directory,
    display_document_summary,
    inspect_sample_document,
)
from rich.console import Console

console = Console()

# ---- AYAR: PDF klasörünün yolu ----
PDF_DIRECTORY = "data/raw_pdfs"


def test_pdf_loading():
    """Aşama 1'in tüm fonksiyonlarını sırayla test eder."""

    console.print("\n" + "="*60, style="bold blue")
    console.print("  🏥 Medical RAG Assistant - Aşama 1 Testi", style="bold blue")
    console.print("="*60 + "\n", style="bold blue")

    # TEST 1: Belgeleri yükle
    console.print("[bold]TEST 1: PDF Yükleme[/bold]")
    try:
        documents = load_documents_from_directory(PDF_DIRECTORY)
        console.print(f"  → [green]BAŞARILI:[/green] {len(documents)} sayfa yüklendi\n")
    except FileNotFoundError as e:
        console.print(f"  → [red]HATA:[/red] {e}\n")
        return
    except ValueError as e:
        console.print(f"  → [red]HATA:[/red] {e}\n")
        return

    # TEST 2: Özet tablosu
    console.print("[bold]TEST 2: Belge Özeti[/bold]")
    display_document_summary(documents)

    # TEST 3: İlk belgeyi incele
    console.print("\n[bold]TEST 3: İlk Sayfayı İncele[/bold]")
    inspect_sample_document(documents, doc_index=0)

    # TEST 4: Metadata doğrulama
    console.print("[bold]TEST 4: Metadata Doğrulama[/bold]")
    hatali_doc = []
    for i, doc in enumerate(documents):
        if "source" not in doc.metadata:
            hatali_doc.append(i)
        if "page" not in doc.metadata:
            hatali_doc.append(i)

    if hatali_doc:
        console.print(f"  → [red]UYARI:[/red] {len(hatali_doc)} belgede metadata eksik!")
    else:
        console.print(f"  → [green]BAŞARILI:[/green] Tüm {len(documents)} belgede metadata mevcut ✅\n")

    # ÖZET
    console.print("="*60, style="bold green")
    console.print("  ✅ AŞAMA 1 TAMAMLANDI!", style="bold green")
    console.print(f"  Toplam {len(documents)} sayfa hazır, Aşama 2'ye geçebiliriz.", style="bold green")
    console.print("="*60 + "\n", style="bold green")


if __name__ == "__main__":
    test_pdf_loading()
