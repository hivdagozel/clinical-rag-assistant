"""
=============================================================
Test: API Entegrasyon ve Entity Extraction Testi
=============================================================
Bu script aşağıdakileri test eder:
  1. Entity Extraction — farklı sorgu formatları
  2. API bağlantısı — server çalışıyor mu?
  3. Tam pipeline — sorgudan Document'e
=============================================================
"""

import sys
from pathlib import Path

# Proje kökünü path'e ekle
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule

console = Console()


def test_entity_extraction():
    """Farklı sorgu formatlarında ilaç adı çıkarmayı test eder."""
    from src.medicine_api_client import extract_medicine_name

    test_cases = [
        ("Parol ne işe yarar?", "Parol"),
        ("Augmentin 1000 mg kullanım şekli nedir?", "Augmentin"),
        ("ASPIRIN günde kaç tane alınır?", "ASPIRIN"),
        ("\"Cipro XR\" nedir?", "Cipro XR"),
        ("ibuprofen yan etkileri nelerdir?", "ibuprofen"),
        ("Bu ilaç nasıl kullanılır?", None),  # İlaç adı yok
    ]

    table = Table(
        title="Entity Extraction Test Sonuclari",
        show_header=True,
        header_style="bold magenta"
    )
    table.add_column("Sorgu", style="cyan", no_wrap=False)
    table.add_column("Beklenen", style="yellow")
    table.add_column("Bulunan", style="green")
    table.add_column("Durum", justify="center")

    passed = 0
    for query, expected in test_cases:
        result = extract_medicine_name(query)
        # Case-insensitive karşılaştırma
        if expected is None:
            success = result is None
        else:
            success = result is not None and expected.lower() in result.lower()

        status = "[bold green]GECTI[/bold green]" if success else "[bold red]KALDI[/bold red]"
        if success:
            passed += 1
        table.add_row(
            query[:50],
            str(expected),
            str(result),
            status
        )

    console.print("\n")
    console.print(table)
    console.print(f"\n[bold]Sonuc: {passed}/{len(test_cases)} test gecti[/bold]")
    return passed


def test_api_connection():
    """API bağlantısını test eder."""
    from src.medicine_api_client import check_api_health, MEDICINE_API_URL

    console.print(Rule("[bold cyan]API Baglanti Testi[/bold cyan]"))
    console.print(f"[dim]Hedef: {MEDICINE_API_URL}[/dim]\n")

    is_healthy = check_api_health()

    if is_healthy:
        console.print(Panel(
            f"[bold green]API ERISEBILIR VE SAGLIKLI![/bold green]\n"
            f"URL: {MEDICINE_API_URL}",
            border_style="green"
        ))
    else:
        console.print(Panel(
            f"[bold yellow]API SIMDILIK ERISEMIYOR[/bold yellow]\n\n"
            f"Bu normal — API sunucusu balatilmamis olabilir.\n\n"
            f"[dim]Balatmak icin:\n"
            f"1. git clone https://github.com/tugcantopaloglu/turkish-medicine-api\n"
            f"2. cd turkish-medicine-api\n"
            f"3. npm install\n"
            f"4. npm run download   (ilk kez)\n"
            f"5. npm start[/dim]",
            border_style="yellow"
        ))

    return is_healthy


def test_full_pipeline(query: str = "Parol ne işe yarar?"):
    """Tam Hybrid RAG pipeline'ını test eder."""
    console.print(Rule("[bold cyan]Tam Pipeline Testi[/bold cyan]"))
    console.print(f"Test sorgusu: [italic]'{query}'[/italic]\n")

    from src.medicine_api_client import get_medicine_context, display_api_documents

    documents = get_medicine_context(query, limit=3)

    if documents:
        display_api_documents(documents)

        # Document içeriğini göster
        console.print(Rule("[dim]Ornek Document icerigi[/dim]"))
        console.print("[bold]page_content (LLM'e gidecek metin):[/bold]")
        console.print(f"[dim]{documents[0].page_content}[/dim]")
        console.print("\n[bold]metadata:[/bold]")
        for k, v in documents[0].metadata.items():
            console.print(f"  [cyan]{k}[/cyan]: {v}")
    else:
        console.print("[yellow]Pipeline testi: API erisilemedigi icin belge gelemedi.[/yellow]")
        console.print("[dim]API sunucusunu balatince tam sonucu gorebilirsiniz.[/dim]")

    return documents


def main():
    console.print(Panel.fit(
        "[bold cyan]Medical RAG Assistant[/bold cyan]\n"
        "[dim]API Entegrasyon Test Suite[/dim]",
        border_style="cyan"
    ))

    # Test 1: Entity Extraction
    console.print(Rule("[bold]TEST 1: Entity Extraction[/bold]"))
    passed = test_entity_extraction()

    # Test 2: API Bağlantısı
    console.print(f"\n")
    api_ok = test_api_connection()

    # Test 3: Tam Pipeline (sadece API çalışıyorsa anlamlı)
    console.print(f"\n")
    test_full_pipeline("Parol ne işe yarar?")

    # Özet
    console.print(f"\n")
    console.print(Panel.fit(
        f"[bold]Test Ozeti[/bold]\n\n"
        f"Entity Extraction : [{'green' if passed >= 4 else 'red'}]{'Basarili' if passed >= 4 else 'Iyilestirme Gerekli'}[/{'green' if passed >= 4 else 'red'}]\n"
        f"API Baglantisi    : [{'green' if api_ok else 'yellow'}]{'Aktif' if api_ok else 'Pasif (API baslatilmadi)'}[/{'green' if api_ok else 'yellow'}]\n\n"
        f"[dim]API'yi baslatmak icin README'ye bakin.[/dim]",
        border_style="cyan"
    ))


if __name__ == "__main__":
    main()
