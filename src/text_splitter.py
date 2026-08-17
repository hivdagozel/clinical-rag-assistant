"""
=============================================================
Aşama 2: Metin Bölme (Text Splitting / Chunking) Modülü
=============================================================

Bu modülün tek sorumluluğu:
  - Aşama 1'den gelen büyük Document'leri alıp
  - Anlamlı, örtüşen küçük parçalara (chunk) bölmek
  - Her chunk'ın kaynak bilgisini (metadata) korumak

Teorik Not — Neden Chunking?
  Vektör veritabanları, metinleri sayısal vektörlere dönüştürür.
  Bu işlem her metin parçası için ayrı yapılır. Eğer bölmeden
  devam etsek, 5 sayfalık bir PDF tek bir vektör noktası olur →
  arama hassasiyeti düşer.

  Chunk boyutu seçimi kritiktir:
    Çok küçük (< 200 karakter): Bağlam kaybolur, anlamsız parçalar
    Çok büyük (> 2000 karakter): Vektör araması kaba kalır
    Optimal (500-1200 karakter): Hem anlam hem hassasiyet korunur

Neden RecursiveCharacterTextSplitter?
  LangChain'in en akıllı splitter'ıdır. Sıradüzensel olarak keser:
    1. Önce paragraf sonu (\n\n) dener
    2. Olmazsa satır sonu (\n)
    3. Olmazsa boşluk ( )
    4. Son çare: karakter karakter
  Böylece cümle ortasında bölmekten kaçınır.

Overlap (Örtüşme) Neden Gerekli?
  "A ilacını B ile birlikte kullanmayın" cümlesi chunk sınırına
  denk gelirse, "A ilacını B ile birlikte" bir chunk'ta,
  "kullanmayın" başka bir chunk'ta kalır. Overlap bunu önler.
=============================================================
"""

import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from src.config import settings
from langchain_core.documents import Document
class SimpleRecursiveSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=200, separators=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

    def _split_text(self, text: str, separators: list) -> list:
        if len(text) <= self.chunk_size:
            return [text]
        if not separators:
            return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]

        separator = separators[0]
        if separator == "":
            splits = list(text)
        else:
            splits = text.split(separator)

        chunks = []
        current_chunk = []
        current_length = 0

        for split in splits:
            split_len = len(split) + (len(separator) if separator != "" else 0)

            if split_len > self.chunk_size:
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                sub_splits = self._split_text(split, separators[1:])
                chunks.extend(sub_splits)
            elif current_length + split_len > self.chunk_size:
                if current_chunk:
                    chunks.append(separator.join(current_chunk))

                # Backtrack for overlap
                overlap_size = self.chunk_overlap
                overlap_chunk = []
                overlap_len = 0
                for prev in reversed(current_chunk):
                    prev_len = len(prev) + (len(separator) if separator != "" else 0)
                    if overlap_len + prev_len <= overlap_size:
                        overlap_chunk.insert(0, prev)
                        overlap_len += prev_len
                    else:
                        break
                current_chunk = overlap_chunk + [split]
                current_length = overlap_len + split_len
            else:
                current_chunk.append(split)
                current_length += split_len

        if current_chunk:
            chunks.append(separator.join(current_chunk))

        return chunks

    def split_documents(self, documents: list) -> list:
        from langchain_core.documents import Document
        split_docs = []
        for doc in documents:
            text = doc.page_content
            chunks = self._split_text(text, self.separators)
            for chunk in chunks:
                split_docs.append(Document(page_content=chunk, metadata=doc.metadata.copy()))
        return split_docs

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns

load_dotenv()
console = Console()

# --- Yapılandırma (.env'den okunur, varsayılanlar tıbbi metinler için optimize edilmiş) ---
CHUNK_SIZE = settings.chunk_size
CHUNK_OVERLAP = settings.chunk_overlap


def create_text_splitter(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> SimpleRecursiveSplitter:
    """
    Tıbbi metinler için optimize edilmiş text splitter oluşturur.

    Özelleştirme:
      Türkçe metinlerde bölme sırası → paragraf → madde işareti → satır → boşluk

    Args:
        chunk_size:    Her chunk'ın maksimum karakter sayısı
        chunk_overlap: Ardışık chunk'lar arasındaki örtüşme

    Returns:
        Yapılandırılmış SimpleRecursiveSplitter
    """
    separators = [
        "\n\n",    # Paragraf sonu (en güçlü bölme noktası)
        "\n",      # Satır sonu
        ". ",      # Cümle sonu (Türkçe)
        "! ",      # Ünlem
        "? ",      # Soru işareti
        "; ",      # Noktalı virgül
        ", ",      # Virgül
        " ",       # Boşluk
        "",        # Son çare: karakter karakter
    ]

    return SimpleRecursiveSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )


def split_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    min_chunk_length: int = 50,  # Bu kadar kısa chunk'ları at
) -> List[Document]:
    """
    Document listesini chunk'lara böler.

    Her chunk yeni bir Document nesnesidir. Orijinal belgenin
    metadata'sı korunur + chunk numarası eklenir.

    Args:
        documents:         Aşama 1'den gelen Document listesi
        chunk_size:        Maksimum chunk boyutu (karakter)
        chunk_overlap:     Chunk'lar arası örtüşme (karakter)
        min_chunk_length:  Bu uzunluğun altındaki chunk'lar atılır

    Returns:
        List[Document]: Bölünmüş chunk'lar (orijinalden çok daha fazla)
    """
    if not documents:
        raise ValueError("Bölünecek Document listesi boş!")

    console.print(f"\n[bold cyan]Metin Bolme Basliyor...[/bold cyan]")
    console.print(f"  Parametre: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    console.print(f"  Girdi: {len(documents)} Document\n")

    splitter = create_text_splitter(chunk_size, chunk_overlap)

    # LangChain'in split_documents metodu:
    #   - Her Document'ı böler
    #   - Orijinal metadata'yı kopyalar
    #   - Yeni Document nesneleri döndürür
    raw_chunks = splitter.split_documents(documents)

    # --- Kalite Filtresi: Çok kısa chunk'ları temizle ---
    # Sadece sayfa numarası veya başlık içeren chunk'lar işe yaramaz
    filtered_chunks = [
        chunk for chunk in raw_chunks
        if len(chunk.page_content.strip()) >= min_chunk_length
    ]

    removed = len(raw_chunks) - len(filtered_chunks)

    # --- Her chunk'a sıra numarası ekle ---
    # Bu, debugging ve kaynak gösterme için çok işe yarar
    for i, chunk in enumerate(filtered_chunks):
        chunk.page_content = chunk.page_content.strip()
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_total"] = len(filtered_chunks)
        # chunk'ın kaynak belgesindeki yaklaşık konumu
        chunk.metadata["chunk_size_actual"] = len(chunk.page_content)

    console.print(f"[green]Bolme tamamlandi:[/green]")
    console.print(f"  {len(documents)} Document -> {len(filtered_chunks)} chunk")
    if removed > 0:
        console.print(f"  [dim]{removed} cok kisa chunk temizlendi[/dim]")
    console.print(
        f"  Ortalama chunk boyutu: "
        f"{sum(len(c.page_content) for c in filtered_chunks) // max(len(filtered_chunks),1)} karakter"
    )

    return filtered_chunks


def split_api_documents(
    api_documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Turkish Medicine API'den gelen Document'leri böler.

    API belgeleri zaten kısa ve yapılandırılmış olduğundan
    genellikle tek chunk olarak kalır. Ama büyük API yanıtları
    için yine de bölme uygulanır.

    Args:
        api_documents: medicine_api_client'ten gelen Document'ler
        chunk_size:    Chunk boyutu
        chunk_overlap: Örtüşme

    Returns:
        Bölünmüş Document'ler (API source_type korunur)
    """
    if not api_documents:
        return []

    # API belgeleri için daha büyük chunk boyutu kullan
    # (yapılandırılmış veri bütünlüğü önemli)
    api_chunk_size = max(chunk_size, 800)
    splitter = create_text_splitter(api_chunk_size, chunk_overlap=50)

    chunks = splitter.split_documents(api_documents)

    # API kaynak bilgisini koru
    for chunk in chunks:
        if "source_type" not in chunk.metadata:
            chunk.metadata["source_type"] = "api"

    return chunks


# ─────────────────────────────────────────────
# Görselleştirme Fonksiyonları
# ─────────────────────────────────────────────

def display_chunk_summary(chunks: List[Document]) -> None:
    """
    Chunk'ların istatistiksel özetini terminalde gösterir.

    Args:
        chunks: split_documents'ten dönen chunk listesi
    """
    if not chunks:
        console.print("[red]Gösterilecek chunk yok.[/red]")
        return

    # Kaynak bazında istatistik
    source_stats: dict = {}
    sizes = []
    for chunk in chunks:
        source = Path(chunk.metadata.get("source", "Bilinmiyor")).name
        size = len(chunk.page_content)
        sizes.append(size)

        if source not in source_stats:
            source_stats[source] = {"adet": 0, "toplam_karakter": 0, "min": size, "max": size}

        source_stats[source]["adet"] += 1
        source_stats[source]["toplam_karakter"] += size
        source_stats[source]["min"] = min(source_stats[source]["min"], size)
        source_stats[source]["max"] = max(source_stats[source]["max"], size)

    # Genel istatistik tablosu
    table = Table(
        title=f"Chunk Analizi ({len(chunks)} toplam chunk)",
        show_header=True,
        header_style="bold magenta"
    )
    table.add_column("Kaynak Belge", style="cyan")
    table.add_column("Chunk Sayisi", justify="center", style="green")
    table.add_column("Ort. Boyut", justify="right", style="yellow")
    table.add_column("Min", justify="right", style="blue")
    table.add_column("Max", justify="right", style="red")

    for source, stats in source_stats.items():
        avg = stats["toplam_karakter"] // stats["adet"]
        table.add_row(
            source[:40],
            str(stats["adet"]),
            f"{avg} kar.",
            f"{stats['min']} kar.",
            f"{stats['max']} kar.",
        )

    console.print("\n")
    console.print(table)

    # Genel özet
    avg_size = sum(sizes) // len(sizes)
    console.print(f"\n[bold]Genel Ozet:[/bold]")
    console.print(f"  Toplam chunk    : {len(chunks)}")
    console.print(f"  Ort. boyut      : {avg_size} karakter")
    console.print(f"  En kucuk chunk  : {min(sizes)} karakter")
    console.print(f"  En buyuk chunk  : {max(sizes)} karakter")


def inspect_sample_chunks(chunks: List[Document], count: int = 3) -> None:
    """
    İlk birkaç chunk'ı detaylı gösterir.

    Amaç: Chunk'ların anlamlı noktalarda bölündüğünü doğrulamak.

    Args:
        chunks: Chunk listesi
        count:  Gösterilecek chunk sayısı
    """
    console.print(f"\n[bold]Ornek Chunk Incelemesi (ilk {count} chunk):[/bold]\n")

    for i, chunk in enumerate(chunks[:count]):
        source = Path(chunk.metadata.get("source", "?")).name
        page = chunk.metadata.get("page", "?")
        c_index = chunk.metadata.get("chunk_index", i)
        c_size = len(chunk.page_content)

        # İçeriğin baş ve sonunu göster
        content = chunk.page_content
        preview = content[:200] + ("..." if len(content) > 200 else "")
        ending = ("..." + content[-100:]) if len(content) > 300 else ""

        panel_content = (
            f"[dim]Kaynak: {source} | Sayfa: {page} | "
            f"Boyut: {c_size} karakter | Index: #{c_index}[/dim]\n\n"
            f"[white]{preview}[/white]"
            + (f"\n[dim]{ending}[/dim]" if ending else "")
        )

        console.print(Panel(
            panel_content,
            title=f"[bold green]Chunk #{i+1}[/bold green]",
            border_style="dim green"
        ))


def visualize_overlap(chunks: List[Document], chunk_idx: int = 0) -> None:
    """
    İki ardışık chunk arasındaki örtüşmeyi (overlap) görselleştirir.
    Bu, overlap parametresinin nasıl çalıştığını anlamak için kullanılır.

    Args:
        chunks:     Chunk listesi
        chunk_idx:  Karşılaştırılacak ilk chunk'ın indeksi
    """
    if len(chunks) < chunk_idx + 2:
        console.print("[yellow]Overlap gösterimi için en az 2 chunk gerekli.[/yellow]")
        return

    chunk_a = chunks[chunk_idx]
    chunk_b = chunks[chunk_idx + 1]

    content_a = chunk_a.page_content
    content_b = chunk_b.page_content

    # Örtüşen kısmı bul (chunk_a'nın sonu == chunk_b'nin başı)
    overlap_text = ""
    for end_pos in range(min(len(content_a), 300), 0, -1):
        candidate = content_a[-end_pos:]
        if content_b.startswith(candidate):
            overlap_text = candidate
            break

    console.print(f"\n[bold]Overlap Gorsellestirme (Chunk #{chunk_idx} ve #{chunk_idx+1}):[/bold]")
    console.print(f"[dim]Bulunan ortusme: {len(overlap_text)} karakter[/dim]\n")

    console.print(Panel(
        f"...{content_a[-200:]}",
        title=f"[cyan]Chunk #{chunk_idx} SONU[/cyan]",
        border_style="cyan"
    ))
    console.print(Panel(
        f"{content_b[:200]}...",
        title=f"[green]Chunk #{chunk_idx+1} BASI[/green]",
        border_style="green"
    ))

    if overlap_text:
        console.print(Panel(
            f"[bold yellow]{overlap_text}[/bold yellow]",
            title="[yellow]ORTUSEN KISIM[/yellow]",
            border_style="yellow"
        ))
    else:
        console.print("[dim]Oturtme metni bulunamadi (beklenti farklı kaynaklarda overlapsiz bolunme)[/dim]")
