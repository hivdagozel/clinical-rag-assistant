"""
=============================================================
Aşama 1: Veri Girişi (Ingestion) - PDF Okuma Modülü
=============================================================

Bu modülün tek sorumluluğu:
  - .env dosyasından PDF klasör yolunu okumak
  - Belirtilen klasör(ler)deki tüm PDF dosyalarını bulmak
  - Alt klasörleri de tarama seçeneği (PDF_RECURSIVE=true)
  - Birden fazla klasörü birleştirme desteği (PDF_EXTRA_DIRS)
  - Her sayfayı ayrı bir Document nesnesi olarak yüklemek
  - Her Document'e kaynak bilgisi (metadata) eklemek

Teorik Not:
  LangChain'in "Document" nesnesi iki parçadan oluşur:
    1. page_content : Sayfanın ham metni (str)
    2. metadata     : Kaynak bilgileri (dict) → {"source": "...", "page": 2}
=============================================================
"""

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from pypdf import PdfReader
from src.metadata_loader import get_metadata, normalize_drug_name
from src.config import PROJECT_ROOT, KT_PDF_DIR, KUB_PDF_DIR, indexed_pdf_directories
from rich.console import Console
from rich.table import Table
from rich import print as rprint

# .env dosyasını yükle (proje kökünden)
load_dotenv()

# Terminalde renkli çıktı için Rich kütüphanesi
console = Console()


def _resolve_path(path_str: str, project_root: Path) -> Path:
    """
    Verilen yolu mutlak hale getirir.
    Eğer göreceli bir yol verilmişse (örn: "data/raw_pdfs"),
    proje kök dizinine göre çözümlenir.
    """
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def get_pdf_directories() -> List[Path]:
    """
    .env dosyasından PDF klasör yollarını okur ve çözümler.

    Desteklenen .env değişkenleri:
      PDF_DIR         → Ana PDF klasörü (tek yol)
      PDF_EXTRA_DIRS  → Ek klasörler (noktalı virgülle ayrılmış)

    Returns:
        List[Path]: Taranacak klasörlerin listesi
    """
    return list(indexed_pdf_directories())


def load_documents_from_directory(pdf_dir: str = None) -> List[Document]:
    """
    .env'den veya parametre olarak verilen klasörden tüm PDF'leri yükler.

    Args:
        pdf_dir: PDF klasörü (None ise .env'den okunur)

    Returns:
        List[Document]: Her sayfa için ayrı bir LangChain Document nesnesi listesi

    Raises:
        FileNotFoundError: Hiçbir klasör bulunamazsa
        ValueError: Hiçbir klasörde PDF dosyası yoksa
    """
    recursive = os.getenv("PDF_RECURSIVE", "true").lower() == "true"

    # Hangi klasörler taranacak?
    if pdf_dir is not None:
        # Manuel yol verilmişse sadece onu kullan
        dirs_to_scan = [_resolve_path(pdf_dir, Path.cwd())]
    else:
        dirs_to_scan = get_pdf_directories()

    console.print(f"\n[bold cyan]Taranacak Klasor Sayisi:[/bold cyan] {len(dirs_to_scan)}")
    console.print(f"[bold cyan]Alt Klasorler:[/bold cyan] {'Dahil' if recursive else 'Haric'}")

    all_documents: List[Document] = []
    total_pdfs_found = 0

    for pdf_path in dirs_to_scan:
        # --- Hata Kontrolü: Klasör var mı? ---
        if not pdf_path.exists():
            console.print(f"[yellow]Klasor bulunamadi, atlaniyor: {pdf_path}[/yellow]")
            continue

        # Kaç PDF var?
        glob_pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_files = list(pdf_path.glob(glob_pattern))

        if not pdf_files:
            console.print(f"[yellow]PDF bulunamadi: {pdf_path}[/yellow]")
            continue

        total_pdfs_found += len(pdf_files)
        console.print(f"\n  [cyan]{pdf_path}[/cyan]")
        console.print(f"  [green]   {len(pdf_files)} PDF dosyasi bulundu[/green]")

        # --- PDF Yükleme ---
        console.print(f"  [yellow]   Yukleniyor...[/yellow]")

        docs = []
        expected_type = "KÜB" if pdf_path.resolve() == KUB_PDF_DIR.resolve() else "KT"
        for pdf_file in sorted(pdf_files):
            try:
                reader = PdfReader(str(pdf_file))
                file_docs = [
                    Document(page_content=page.extract_text() or "", metadata={"page": page_number})
                    for page_number, page in enumerate(reader.pages)
                ]
            except Exception as exc:
                console.print(f"[red]PDF okunamadı, atlandı: {pdf_file.name}: {exc}[/red]")
                continue
            for doc in file_docs:
                doc.metadata["source"] = str(pdf_file.resolve())
                doc.metadata["source_type"] = "pdf"
                doc.metadata["document_type"] = expected_type
            docs.extend(file_docs)

        # Metadata Zenginlestirme
        for doc in docs:
            source_path = doc.metadata.get("source", "")
            if source_path:
                titck_meta = get_metadata(source_path)
                if titck_meta:
                    for k, v in titck_meta.items():
                        if k not in ["source", "page"]:
                            doc.metadata[k] = v
                    doc.metadata["document_type"] = str(doc.metadata.get("document_type", expected_type)).upper()
                    doc.metadata["normalized_drug_name"] = normalize_drug_name(doc.metadata.get("drug_name", ""))
                else:
                    # Metadata bulunamadiysa ufak bir log atabiliriz ama cok gurultu yapmasin
                    pass

        all_documents.extend(docs)
        console.print(f"  [green]   {len(docs)} sayfa yuklendi (Metadata entegre edildi)[/green]")

    # --- Sonuç Kontrolü ---
    if total_pdfs_found == 0 and pdf_dir is not None:
        valid_dirs = [str(d) for d in dirs_to_scan]
        raise ValueError(
            f"Hiçbir klasörde PDF bulunamadı!\n"
            f"Taranan klasörler:\n" + "\n".join(f"  - {d}" for d in valid_dirs) + "\n\n"
            f"Çözüm:\n"
            f"  1. .env dosyasındaki PDF_DIR değerini güncelleyin\n"
            f"  2. Veya PDF dosyalarını data/raw_pdfs/ klasörüne kopyalayın"
        )
    if total_pdfs_found == 0:
        return []

    console.print(
        f"\n[bold green]Toplam {len(all_documents)} sayfa yuklendi "
        f"({total_pdfs_found} PDF dosyasından)[/bold green]"
    )

    return all_documents


def display_document_summary(documents: List[Document]) -> None:
    """
    Yüklenen belgelerin özetini renkli bir tablo olarak terminale yazdırır.

    Args:
        documents: load_documents_from_directory'den dönen Document listesi
    """
    source_stats: dict = {}
    for doc in documents:
        source_name = Path(doc.metadata.get("source", "Bilinmiyor")).name
        if source_name not in source_stats:
            source_stats[source_name] = {"sayfa_sayisi": 0, "toplam_karakter": 0}

        source_stats[source_name]["sayfa_sayisi"] += 1
        source_stats[source_name]["toplam_karakter"] += len(doc.page_content)

    table = Table(
        title="Yuklenen Tibbi Belgeler",
        show_header=True,
        header_style="bold magenta"
    )
    table.add_column("Belge Adi", style="cyan", no_wrap=False)
    table.add_column("Sayfa Sayisi", justify="center", style="green")
    table.add_column("Toplam Karakter", justify="right", style="yellow")
    table.add_column("Ort. Karakter/Sayfa", justify="right", style="blue")

    for source_name, stats in source_stats.items():
        avg_chars = stats["toplam_karakter"] // stats["sayfa_sayisi"]
        table.add_row(
            source_name,
            str(stats["sayfa_sayisi"]),
            f"{stats['toplam_karakter']:,}",
            f"{avg_chars:,}"
        )

    console.print("\n")
    console.print(table)


def inspect_sample_document(documents: List[Document], doc_index: int = 0) -> None:
    """
    Belirli bir Document'in içeriğini ve metadata'sını inceleme amaçlı gösterir.
    Geliştirme sırasında veri kalitesini doğrulamak için kullanılır.

    Args:
        documents: Document listesi
        doc_index: İncelenecek belgenin indeksi (varsayılan: 0)
    """
    if doc_index >= len(documents):
        console.print(f"[red]Hata: {doc_index}. belge yok (toplam: {len(documents)})[/red]")
        return

    doc = documents[doc_index]
    source_name = Path(doc.metadata.get("source", "?")).name
    page_num = doc.metadata.get("page", 0) + 1

    console.print(f"\n[bold]Ornek Belge Incelemesi (Indeks: {doc_index})[/bold]")
    console.print(f"  [cyan]Kaynak  :[/cyan] {source_name}")
    console.print(f"  [cyan]Sayfa   :[/cyan] {page_num}")
    console.print(f"  [cyan]Karakter:[/cyan] {len(doc.page_content):,}")
    console.print(f"  [cyan]Metadata:[/cyan] {doc.metadata}")
    console.print("\n  [bold yellow]--- Icerik Onizlemesi (ilk 500 karakter) ---[/bold yellow]")
    console.print(f"  {doc.page_content[:500]}...")
    console.print("  [bold yellow]--- Onizleme Sonu ---[/bold yellow]\n")
