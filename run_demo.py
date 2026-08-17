"""
=============================================================
Medical RAG Assistant - Demo Çalıştırıcı
=============================================================
Bu script:
  1. Bağımlılıkları kontrol eder
  2. data/raw_pdfs klasörüne örnek bir PDF oluşturur (eğer yoksa)
  3. document_loader.py'yi çalıştırır ve sonuçları gösterir
=============================================================
"""

import sys
import os
from pathlib import Path

# Proje kökünü sys.path'e ekle
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rich.console import Console
from rich.panel import Panel
from rich import print as rprint

console = Console()

def create_sample_pdf():
    """
    Test için örnek bir tıbbi PDF oluşturur.
    Gerçek kullanımda buraya kendi PDF'lerinizi ekleyeceksiniz.
    """
    pdf_dir = project_root / "data" / "raw_pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    sample_pdf_path = pdf_dir / "ornek_ilac_prospektusu.pdf"

    if sample_pdf_path.exists():
        console.print("[yellow]ℹ️  Örnek PDF zaten mevcut, atlanıyor...[/yellow]")
        return

    try:
        from fpdf import FPDF
    except ImportError:
        console.print("[yellow]⚠️  fpdf2 bulunamadı, pypdf ile minimal PDF oluşturuluyor...[/yellow]")
        _create_minimal_pdf(sample_pdf_path)
        return

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)

    content_lines = [
        "ORNEK ILAC PROSPEKTUSU",
        "",
        "Ilac Adi: Parasetamol 500mg Tablet",
        "",
        "ENDIKASYONLAR:",
        "Hafif ve orta siddetli agri ve ateste kullanilir.",
        "Bas agrisi, dis agrisi, kas agrisi, eklem agrisi,",
        "soguk alginligi ve gribal enfeksiyonlara bagli agri",
        "ve ateste etkilidir.",
        "",
        "DOZAJ VE KULLANIM SEKLI:",
        "Yetiskinler ve 12 yas uzeri cocuklar icin:",
        "Gunde 3-4 kez, her dozda 500-1000 mg alinir.",
        "Dozlar arasi en az 4-6 saat olmalidir.",
        "Gunluk maksimum doz: 4000 mg.",
        "",
        "KONTRENDIKASYONLAR:",
        "Parasetamole karsi bilinen asiri duyarlilik.",
        "Agir karaciger yetmezligi.",
        "",
        "UYARILAR:",
        "Karaciger veya bobrek hastaligi olanlarda dikkatli kullaniniz.",
        "Alkol kullananlar dikkatli olmalidir.",
        "Ayni anda baska parasetamol iceren urunler kullanmayiniz.",
        "",
        "YAN ETKILER:",
        "Nadir: Alerjik reaksiyonlar, deri dokuntuleri.",
        "Cok nadir: Ciddi karaciger hasari (asiri dozda).",
        "",
        "SAKLAMA KOSULLARI:",
        "25 derecenin altinda, kuru bir yerde saklayin.",
        "Cocuklarin ulasamayacagi yerlerde bulundurun.",
    ]

    for line in content_lines:
        pdf.cell(0, 8, text=line, ln=True)

    pdf.output(str(sample_pdf_path))
    console.print(f"[green]✅ Örnek PDF oluşturuldu: {sample_pdf_path.name}[/green]")


def _create_minimal_pdf(output_path: Path):
    """fpdf2 yoksa pypdf ile minimal bir PDF oluşturur."""
    # En basit geçerli PDF formatı
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj

2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj

3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj

4 0 obj
<< /Length 200 >>
stream
BT
/F1 12 Tf
50 750 Td
(ORNEK ILAC PROSPEKTUSU) Tj
0 -20 Td
(Ilac Adi: Parasetamol 500mg Tablet) Tj
0 -20 Td
(ENDIKASYONLAR: Hafif ve orta siddetli agri ve ateste kullanilir.) Tj
0 -20 Td
(DOZAJ: Gunluk maksimum doz 4000 mg dir.) Tj
0 -20 Td
(KONTRENDIKASYONLAR: Parasetamole asiri duyarlilik.) Tj
ET
endstream
endobj

5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj

xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000266 00000 n
0000000516 00000 n

trailer
<< /Size 6 /Root 1 0 R >>
startxref
605
%%EOF"""

    output_path.write_bytes(pdf_content)
    console.print(f"[green]✅ Minimal test PDF oluşturuldu: {output_path.name}[/green]")


def main():
    console.print(Panel.fit(
        "[bold cyan]🏥 Medical RAG Assistant[/bold cyan]\n"
        "[dim]Aşama 1: Belge Yükleme Demo'su[/dim]",
        border_style="cyan"
    ))

    # 1. Örnek PDF oluştur
    console.print("\n[bold]📝 Adım 1: Test PDF hazırlanıyor...[/bold]")
    create_sample_pdf()

    # 2. Document Loader'ı çalıştır
    console.print("\n[bold]📚 Adım 2: Document Loader çalıştırılıyor...[/bold]")
    try:
        from src.document_loader import (
            load_documents_from_directory,
            display_document_summary,
            inspect_sample_document,
        )

        pdf_dir = str(project_root / "data" / "raw_pdfs")
        documents = load_documents_from_directory(pdf_dir)
        display_document_summary(documents)
        inspect_sample_document(documents, doc_index=0)

        console.print(Panel.fit(
            f"[bold green]✅ Demo başarıyla tamamlandı![/bold green]\n\n"
            f"[white]Toplam {len(documents)} sayfa yüklendi.[/white]\n"
            f"[dim]Gerçek tıbbi PDF'lerinizi data/raw_pdfs/ klasörüne ekleyebilirsiniz.[/dim]",
            border_style="green"
        ))

    except ImportError as e:
        console.print(f"\n[red]❌ Import Hatası: {e}[/red]")
        console.print("[yellow]💡 Çözüm: pip install -r requirements.txt[/yellow]")
    except FileNotFoundError as e:
        console.print(f"\n[red]❌ Dosya Hatası: {e}[/red]")
    except ValueError as e:
        console.print(f"\n[red]❌ Değer Hatası: {e}[/red]")
    except Exception as e:
        console.print(f"\n[red]❌ Beklenmedik hata: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
