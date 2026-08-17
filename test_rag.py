import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.rag_chain import MedicalRAGChain, initialize_database_if_empty
from rich.console import Console

console = Console()

def main():
    console.print("[bold yellow]1. Veritabanı ilklendiriliyor...[/bold yellow]")
    initialize_database_if_empty()
    
    console.print("[bold yellow]2. RAG Zinciri yukleniyor...[/bold yellow]")
    rag = MedicalRAGChain()
    
    query = "Ateşim var ve başım çok ağrıyor, hangi ilacı almalıyım?"
    console.print(f"[bold yellow]3. Soru soruluyor:[/bold yellow] '{query}'")
    
    response = rag.ask(query)
    
    console.print("\n[bold green]🤖 RAG Yanıtı:[/bold green]")
    console.print(response["answer"])
    console.print("\n[bold green]Kaynaklar:[/bold green]")
    console.print(response["sources"])
    console.print("\n[bold green]İstatistikler:[/bold green]")
    console.print(response["retrieval_stats"])

if __name__ == "__main__":
    main()
