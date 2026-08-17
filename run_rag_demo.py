"""
=============================================================
🏥 Medical RAG Assistant - Uçtan Uca RAG Demo Çalıştırıcı
=============================================================
Bu script:
  1. Ortam değişkenlerini ve API anahtarını kontrol eder.
  2. Vektör veritabanını (FAISS) kontrol eder ve gerekirse PDF'leri yükler.
  3. Kullanıcıdan etkileşimli soru alıp RAG cevabını üretir.
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
from rich.prompt import Prompt
from rich.table import Table
from rich.rule import Rule
from rich.live import Live
from dotenv import load_dotenv

load_dotenv()
console = Console()

def print_welcome():
    console.print(Panel(
        "[bold cyan]🏥 Medical RAG Assistant - Uçtan Uca Demo 🏥[/bold cyan]\n\n"
        "[white]Bu asistan, tıbbi PDF prospektüsleri ve Turkish Medicine API verilerini kullanarak\n"
        "ilaçlar hakkında doğru, kaynak gösteren ve güvenli Türkçe cevaplar üretir.[/white]\n\n"
        "[dim]Çıkış yapmak için [bold red]exit[/bold red] veya [bold red]quit[/bold red] yazabilirsiniz.[/dim]",
        border_style="cyan"
    ))

def check_env():
    """Çevre değişkenlerini ve Gemini API key'i doğrular."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        console.print(Panel(
            "[bold red]❌ HATA: Google Gemini API Anahtarı Tanımlanmamış! ❌[/bold red]\n\n"
            "Lütfen projenin ana dizininde bulunan [bold yellow].env[/bold yellow] dosyasını açın ve:\n"
            "[bold green]GEMINI_API_KEY=your_actual_gemini_key[/bold green] alanına geçerli API anahtarınızı girin.\n\n"
            "[white]Ücretsiz Gemini API anahtarınızı hemen almak için şu adresi ziyaret edebilirsiniz:[/white]\n"
            "[bold blue]https://aistudio.google.com/[/bold blue]\n\n"
            "[dim]Not: Yerel makinedeki GPU/DLL hatalarını baypas etmek amacıyla sistem tamamen bulut tabanlı\n"
            "ücretsiz Google Gemini modellerine göre yapılandırılmıştır.[/dim]",
            title="Yapılandırma Eksik",
            border_style="red"
        ))
        return False
    return True

def ensure_sample_pdf_indexed():
    """Test için en az 1 örnek PDF'in veri tabanına işlendiğinden emin olur."""
    from run_demo import create_sample_pdf
    from src.vector_store import get_db_stats
    
    # 1. Klasör ve örnek PDF oluştur (yoksa)
    create_sample_pdf()
    
    # 2. Veri tabanı durumuna bak, boşsa yükle
    from src.rag_chain import initialize_database_if_empty
    initialize_database_if_empty()

def main():
    print_welcome()
    
    # .env Kontrolü
    if not check_env():
        return
        
    # Veri Tabanı Hazırlığı
    console.print("\n[bold]⚙️ Veritabanı ve Belgeler Kontrol Ediliyor...[/bold]")
    try:
        ensure_sample_pdf_indexed()
    except Exception as e:
        console.print(f"[red]❌ Veritabanı hazırlığında beklenmedik hata: {e}[/red]")
        return
        
    # RAG Zincirini Yükle
    console.print("\n[bold green]🚀 Yapay Zeka RAG Zinciri Başlatılıyor...[/bold green]")
    try:
        from src.rag_chain import MedicalRAGChain
        rag = MedicalRAGChain()
        console.print("[bold green]✅ RAG Sistemi Sorularınızı Cevaplamaya Hazır![/bold green]")
    except Exception as e:
        console.print(f"[red]❌ RAG başlatma hatası: {e}[/red]")
        return

    # Etkileşimli Döngü
    while True:
        try:
            console.print("\n" + "═"*70)
            user_input = Prompt.ask("\n[bold cyan]Sorunuzu yazın[/bold cyan]")
            
            # Çıkış kontrolleri
            if user_input.strip().lower() in ["exit", "quit", "çıkış", "cikis"]:
                console.print("[bold yellow]👋 Program sonlandırılıyor. Sağlıklı günler dileriz![/bold yellow]")
                break
                
            if not user_input.strip():
                continue
                
            # Soruyu yanıtla
            response = rag.ask(user_input)
            
            # Yanıt Paneli
            console.print("\n[bold green]🤖 Cevap:[/bold green]")
            console.print(Panel(
                response["answer"],
                title="Tıbbi Yapay Zeka Yanıtı",
                border_style="green"
            ))
            
            # Kaynaklar Tablosu
            if response["sources"]:
                table = Table(title="Kullanılan Referans Kaynaklar", border_style="dim")
                table.add_column("Kaynak Adı", style="cyan")
                table.add_column("Türü", style="magenta")
                
                # Tekilleştirilmiş kaynaklar
                seen_sources = set()
                for src in response["sources"]:
                    src_name = src["source"]
                    src_type = src["type"].upper()
                    if src_name not in seen_sources:
                        table.add_row(src_name, src_type)
                        seen_sources.add(src_name)
                        
                console.print(table)
            
        except KeyboardInterrupt:
            console.print("\n[bold yellow]👋 Program sonlandırılıyor. Sağlıklı günler dileriz![/bold yellow]")
            break
        except Exception as e:
            console.print(f"[bold red]❌ Bir hata oluştu:[/bold red] {e}")

if __name__ == "__main__":
    main()
