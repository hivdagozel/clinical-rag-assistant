"""
=============================================================
Test: Aşama 2 - Metin Bölme (Chunking) Testi
=============================================================
Bu script:
  1. PDF belgesini yükler
  2. Metin bölme işlemini uygular
  3. Chunk istatistiklerini ve örtüşmeyi (overlap) gösterir
=============================================================
"""

import sys
from pathlib import Path

# Proje kökünü path'e ekle
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

console = Console()

def main():
    console.print(Panel.fit(
        "[bold cyan]Medical RAG Assistant[/bold cyan]\n"
        "[dim]Asama 2: Metin Bolme (Chunking) Testi[/dim]",
        border_style="cyan"
    ))

    # 1. PDF Belgelerini Yükle
    console.print(Rule("[bold]1. PDF Yukleme[/bold]"))
    from src.document_loader import load_documents_from_directory
    
    pdf_dir = project_root / "data" / "raw_pdfs"
    
    # Örnek PDF yoksa run_demo.py'den oluştur
    if not list(pdf_dir.glob("*.pdf")):
        console.print("[yellow]PDF bulunamadi. Ornek PDF olusturuluyor...[/yellow]")
        from run_demo import create_sample_pdf
        create_sample_pdf()
        
    try:
        documents = load_documents_from_directory(str(pdf_dir))
    except Exception as e:
        console.print(f"[red]Hata: {e}[/red]")
        return

    # 2. Metin Bölme (Chunking)
    console.print(Rule("[bold]2. Metin Bolme Islemi[/bold]"))
    from src.text_splitter import (
        split_documents, 
        display_chunk_summary, 
        inspect_sample_chunks, 
        visualize_overlap
    )
    
    # Daha küçük parametrelerle test edelim ki örnek PDF'te nasıl bölündüğünü görebilelim
    chunks = split_documents(documents, chunk_size=200, chunk_overlap=50)
    
    # 3. Sonuçları Göster
    console.print(Rule("[bold]3. Istatistikler ve Sonuclar[/bold]"))
    display_chunk_summary(chunks)
    inspect_sample_chunks(chunks, count=3)
    visualize_overlap(chunks, chunk_idx=0)

if __name__ == "__main__":
    main()
