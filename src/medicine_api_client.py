"""
=============================================================
Hybrid RAG: Turkish Medicine API Client
=============================================================

Bu modülün sorumluluğu:
  1. Kullanıcı sorgusundan ilaç adını çıkarmak  (Entity Extraction)
  2. Turkish Medicine API'ye GET isteği atmak    (API Retrieval)
  3. Dönen JSON'ı LangChain Document'e çevirmek (Document Building)

Kaynak API: https://github.com/tugcantopaloglu/turkish-medicine-api
API Endpointleri:
  GET /api/medicines/search?q={ilaç_adı}         → İlaç arama
  GET /api/medicines/{id}                         → ID ile getir
  GET /api/medicines/filter?field=İlaç Adı&value=X → Alan filtresi

Teorik Not (Entity Extraction):
  "Entity Extraction" (Varlık Çıkarma), ham metinden anlamlı
  bilgileri tespit etme işlemidir. Basit yaklaşımımız:
  - Türk ilaç isimlerine özgü büyük harf kalıplarını tanı
  - Belirli ilaç anahtar kelimelerini ara
  - LLM destekli extraction (gelecek aşama)
=============================================================
"""

import re
import os
import json
import urllib.request
import urllib.parse
import urllib.error
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from src.config import settings
from src.query_analysis import extract_product_name

load_dotenv()

console = Console()

# --- Yapılandırma ---
MEDICINE_API_URL = settings.medicine_api_url
MEDICINE_API_TIMEOUT = settings.medicine_api_timeout

# Türkiye'de yaygın kullanılan ilaç ismi kalıpları
# Bu liste "stop words" değil — ilaç adı ipuçlarıdır
MEDICINE_KEYWORDS = [
    "tablet", "kapsül", "şurup", "ampul", "flakon", "enjeksiyon",
    "krem", "merhem", "damla", "sprey", "inhaler", "jel", "patch",
    "mg", "ml", "mcg", "iu"
]

# Kullanıcı sorgusundan ilaç adı çıkarmaya yardımcı anahtar ifadeler
QUERY_TRIGGERS = [
    "ne işe yarar", "nedir", "ne için kullanılır", "yan etkileri", "dozu",
    "nasıl kullanılır", "endikasyonları", "kontrendikasyonları",
    "hakkında bilgi", "prospektüs", "kullanım", "fiyatı", "içeriği",
    "etken maddesi", "ne zaman kullanılır", "kaç mg", "dozaj"
]

# İlaç adı olarak kabul edilmeyecek yaygın kelimeler (Stop Words)
STOP_MEDICINE_NAMES = {
    "bu", "o", "şu", "ilaç", "ilacı", "ilaçlar", "ilaçları", "hangisi",
    "nedir", "ne", "nasıl", "kim", "nerede", "bu ilaç", "o ilaç", "şu ilaç",
    "hakkında", "bilgi", "prospektüs", "prospektüsü", "etki", "etkisi", "günde",
    "kullanılır", "kullanımı", "kullanım", "şekli"
}


def normalize_text(text: str) -> str:
    """Türkçe karakterleri İngilizce karşılıklarına dönüştürerek metni normalize eder."""
    translation_table = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
    return text.translate(translation_table).lower()

# Marka isimleri ile etken maddeler (jenerik isimler) arasındaki eşleşme haritası
MEDICINE_SYNONYMS = {
    "parol": ["parasetamol", "paracetamol", "acetaminophen"],
    "aspirin": ["asetilsalisilik asit", "asa"],
    "augmentin": ["amoksisilin", "klavulanik asit", "amoxicillin"],
    "coraspin": ["asetilsalisilik asit", "asa"],
    "cipro": ["siprofloksasin", "ciprofloxacin"],
    "lansor": ["lansoprazol", "lansoprazole"],
    "arvales": ["dekstetoprofen", "dexketoprofen"],
    "dolorex": ["diklofenak", "diclofenac"],
    "majezik": ["flurbiprofen"]
}


# ─────────────────────────────────────────────
# 1. Entity Extraction: İlaç Adı Çıkarma
# ─────────────────────────────────────────────

def extract_medicine_name(query: str) -> Optional[str]:
    """
    Kullanıcı sorgusundan ilaç adını çıkarır.

    Yaklaşım (kademeli — önce basit, sonra karmaşık):
      Adım 0: Bilinen ilaç/etken madde sözlük taraması (Dictionary-based match)
      Adım 1: Tırnak içindeki metni ara → "Parol ne işe yarar?" → "Parol"
      Adım 2: Büyük harfle başlayan kelimelerden ilaç olanları bul
      Adım 3: Sorgunun ilk anlamlı kelimesini dene (fallback)

    Args:
        query: Kullanıcının ham sorusu

    Returns:
        Tespit edilen ilaç adı (str) veya None
    """
    query = query.strip()
    product = extract_product_name(query)
    if product:
        return product.title()

    # --- Adım 0: Bilinen ilaç/etken madde sözlük taraması ---
    query_norm = normalize_text(query)
    for key, synonyms in MEDICINE_SYNONYMS.items():
        # Ana marka adını kontrol et (kelime sınırı ile)
        if re.search(r'\b' + re.escape(key) + r'\b', query_norm):
            console.print(f"  [dim]→ Sözlük tespiti (marka): '{key.capitalize()}'[/dim]")
            return key.capitalize()
        # Etken maddeleri kontrol et
        for syn in synonyms:
            if len(syn) > 2 and re.search(r'\b' + re.escape(syn) + r'\b', query_norm):
                console.print(f"  [dim]→ Sözlük tespiti (etken madde): '{syn.capitalize()}'[/dim]")
                return syn.capitalize()

    query_lower = query.lower()

    # --- Adım 1: Tırnak içi kontrol ---
    # "Parol 500" veya 'Augmentin' gibi
    quoted = re.findall(r'["\']([^"\']+)["\']', query)
    if quoted:
        candidate = quoted[0].strip()
        if candidate.lower() not in STOP_MEDICINE_NAMES:
            console.print(f"  [dim]→ Tırnak içi tespit: '{candidate}'[/dim]")
            return candidate

    # --- Adım 2: Büyük harfli token kalıpları ---
    # Türk ilaç isimleri genellikle büyük harfle başlar: Parol, Augmentin, Cipro XR
    # Sorguda tetikleyici kelimeden ÖNCE gelen büyük harfli kelimeleri ara

    # Tetikleyici kelimeden önce ne var?
    for trigger in QUERY_TRIGGERS:
        if trigger in query_lower:
            # Tetikleyicinin öncesindeki kısmı al
            before_trigger = query_lower.index(trigger)
            prefix = query[:before_trigger].strip()
            if prefix:
                # Büyük harfli kelimeleri çek (ilaç adı genelde ilk kelimeler)
                words = prefix.split()
                medicine_words = []
                for word in words:
                    # Büyük harfle başlıyor mu veya tamamı büyük mü?
                    clean = re.sub(r'[^a-zA-ZÇĞİÖŞÜçğışöşü0-9\s\-]', '', word)
                    if clean and (clean[0].isupper() or clean.isupper()):
                        medicine_words.append(clean)
                    else:
                        break  # İlk küçük harfli kelimede dur
                if medicine_words:
                    candidate = " ".join(medicine_words)
                    if candidate.lower() not in STOP_MEDICINE_NAMES:
                        console.print(f"  [dim]→ Tetikleyici öncesi tespit: '{candidate}'[/dim]")
                        return candidate

    # --- Adım 3: Sorgunun tamamından büyük harfli blok ---
    words = query.split()
    medicine_words = []
    for word in words:
        clean = re.sub(r'[^a-zA-ZÇĞİÖŞÜçğışöşü0-9\-]', '', word)
        if clean and (clean[0].isupper() or clean.isupper()) and len(clean) > 1:
            # Sadece ilaç adına ait kelimeleri al
            # "MG", "ML" gibi birim kelimeleri de dahil et
            medicine_words.append(clean)
        elif medicine_words:
            break  # İlk boşluk/küçük harf bloğunda dur

    if medicine_words:
        candidate = " ".join(medicine_words)
        if candidate.lower() not in STOP_MEDICINE_NAMES:
            console.print(f"  [dim]→ İlk büyük harfli blok: '{candidate}'[/dim]")
            return candidate

    # --- Adım 4: Fallback — sorgunun ilk kelimesi ---
    first_word = words[0] if words else None
    if first_word:
        clean = re.sub(r'[^a-zA-ZÇĞİÖŞÜçğışöşü0-9]', '', first_word)
        if len(clean) > 2 and clean.lower() not in STOP_MEDICINE_NAMES:
            console.print(f"  [dim]→ Fallback (ilk kelime): '{clean}'[/dim]")
            return clean

    console.print("  [dim red]→ İlaç adı tespit edilemedi[/dim red]")
    return None


# ─────────────────────────────────────────────
# 2. API Client: Turkish Medicine API'ye İstek
# ─────────────────────────────────────────────

def check_api_health() -> bool:
    """
    API sunucusunun çalışıp çalışmadığını kontrol eder.

    Returns:
        True: API erişilebilir
        False: API kapalı veya hata var
    """
    if not settings.use_medicine_api:
        return False
    try:
        url = f"{MEDICINE_API_URL}/health"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=MEDICINE_API_TIMEOUT) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data.get("status") == "healthy"
    except Exception:
        return False


def search_medicine(medicine_name: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Turkish Medicine API'de ilaç adıyla arama yapar.

    Endpoint: GET /api/medicines/search?q={name}&limit={limit}

    Args:
        medicine_name: Aranacak ilaç adı (ör: "Parol", "Augmentin")
        limit: Döndürülecek maksimum sonuç sayısı

    Returns:
        API'den dönen ilaç kayıtları listesi (her biri dict)

    Raises:
        ConnectionError: API'ye ulaşılamazsa
        ValueError: API hata döndürürse
    """
    # URL parametrelerini encode et (Türkçe karakter desteği)
    params = urllib.parse.urlencode({
        "q": medicine_name,
        "limit": limit,
        "sheet": "AKTİF ÜRÜNLER LİSTESİ"  # Sadece aktif ürünler
    })
    url = f"{MEDICINE_API_URL}/api/medicines/search?{params}"

    console.print(f"  [dim cyan]API: GET {url}[/dim cyan]")

    try:
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "MedicalRAGAssistant/1.0"
            }
        )
        with urllib.request.urlopen(req, timeout=MEDICINE_API_TIMEOUT) as response:
            raw = response.read().decode("utf-8")
            data = json.loads(raw)

            # API yanıt formatı: {"data": [...], "total": N, ...}
            if "data" in data:
                return data["data"]
            elif "error" in data:
                raise ValueError(f"API Hatası: {data['error']}")
            else:
                return []

    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Turkish Medicine API'ye bağlanılamadı: {MEDICINE_API_URL}\n"
            f"Hata: {e}\n\n"
            f"Çözüm:\n"
            f"  1. API repo'sunu clone edin: git clone https://github.com/tugcantopaloglu/turkish-medicine-api\n"
            f"  2. npm install && npm run download && npm start\n"
            f"  3. Veya .env'deki MEDICINE_API_URL değerini güncelleyin"
        ) from e


def get_medicine_by_id(medicine_id: int) -> Optional[Dict[str, Any]]:
    """
    Belirli bir ilaç ID'si ile detaylı bilgi çeker.

    Endpoint: GET /api/medicines/{id}

    Args:
        medicine_id: İlaç veritabanı ID'si

    Returns:
        İlaç bilgisi dict veya None
    """
    url = f"{MEDICINE_API_URL}/api/medicines/{medicine_id}"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=MEDICINE_API_TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


# ─────────────────────────────────────────────
# 3. Document Builder: JSON → LangChain Document
# ─────────────────────────────────────────────

def build_document_from_medicine(medicine: Dict[str, Any], query: str = "") -> Document:
    """
    API'den gelen tek bir ilaç kaydını LangChain Document nesnesine dönüştürür.

    Neden Document? LangChain'in tüm RAG pipeline'ı Document üzerinden çalışır.
    Bu dönüşüm sayesinde API'den gelen veriyi PDF'ten gelen veri ile
    aynı şekilde kullanabiliyoruz — Hybrid RAG'ın özü budur.

    page_content formatı:
      İnsan tarafından okunabilir, zengin metin (LLM bunu okuyacak)

    metadata formatı:
      Makine tarafından işlenebilir yapısal bilgiler

    Args:
        medicine: API'den gelen tek ilaç kaydı (dict)
        query: Kullanıcının orijinal sorusu (metadata için)

    Returns:
        LangChain Document nesnesi
    """
    # API alanları Türkçe isimlerle geliyor (orijinal Excel formatı)
    ilac_adi = medicine.get("İlaç Adı", medicine.get("ilac_adi", "Bilinmiyor"))
    firma = medicine.get("Firma Adı", medicine.get("firma_adi", "Bilinmiyor"))
    barkod = medicine.get("Barkod", medicine.get("barkod", ""))
    atc_kodu = medicine.get("ATC Kodu", medicine.get("atc_kodu", ""))
    atc_adi = medicine.get("ATC Adı", medicine.get("atc_adi", ""))
    recete_turu = medicine.get("Reçete Türü", medicine.get("recete_turu", ""))
    durumu = medicine.get("Durumu", medicine.get("durumu", ""))
    sheet = medicine.get("_sheet", "")
    med_id = medicine.get("id", "")

    # ── page_content: LLM'in okuyacağı zengin metin ──
    content_parts = [
        f"İLAÇ BİLGİ KARTI",
        f"{'='*40}",
        f"İlaç Adı     : {ilac_adi}",
        f"Firma        : {firma}",
        f"ATC Kodu     : {atc_kodu}",
        f"ATC Adı      : {atc_adi}",
        f"Barkod       : {barkod}",
        f"Reçete Türü  : {recete_turu}",
        f"Durumu       : {durumu}",
        f"Kayıt No (ID): {med_id}",
        f"Veri Kaynağı : TITCK - Turkish Medicine API",
        f"{'='*40}",
    ]

    # Ek alanlar varsa ekle (API bazen fazladan alanlar döndürür)
    exclude_keys = {
        "İlaç Adı", "ilac_adi", "Firma Adı", "firma_adi",
        "Barkod", "barkod", "ATC Kodu", "atc_kodu",
        "ATC Adı", "atc_adi", "Reçete Türü", "recete_turu",
        "Durumu", "durumu", "_sheet", "id"
    }
    for key, value in medicine.items():
        if key not in exclude_keys and value:
            content_parts.append(f"{key}: {value}")

    page_content = "\n".join(content_parts)

    # ── metadata: RAG pipeline'ı için yapısal bilgiler ──
    api_search_url = f"{MEDICINE_API_URL}/api/medicines/{med_id}"
    metadata = {
        "source": "Turkish Medicine API (TITCK)",
        "source_type": "api",                          # PDF vs API ayrımı
        "api_url": api_search_url,
        "medicine_name": ilac_adi,
        "medicine_id": med_id,
        "barcode": str(barkod),
        "atc_code": atc_kodu,
        "company": firma,
        "prescription_type": recete_turu,
        "status": durumu,
        "sheet": sheet,
        "original_query": query,
    }

    return Document(page_content=page_content, metadata=metadata)


def build_documents_from_results(
    api_results: List[Dict[str, Any]],
    query: str = ""
) -> List[Document]:
    """
    API'den gelen tüm ilaç listesini Document listesine dönüştürür.

    Args:
        api_results: search_medicine() fonksiyonundan gelen liste
        query: Kullanıcının orijinal sorusu

    Returns:
        List[Document]: Her ilaç için bir Document nesnesi
    """
    documents = []
    for medicine in api_results:
        doc = build_document_from_medicine(medicine, query)
        documents.append(doc)
    return documents


# ─────────────────────────────────────────────
# 4. Ana Orkestrasyon Fonksiyonu
# ─────────────────────────────────────────────

def get_medicine_context(query: str, limit: int = 3) -> List[Document]:
    """
    Kullanıcı sorusundan ilaç bağlamını (context) döndürür.

    Bu fonksiyon tüm pipeline'ı yönetir:
      sorgu → ilaç adı çıkar → API'ye sor → Document'e dönüştür

    Args:
        query: Kullanıcının sorusu (ör: "Parol ne işe yarar?")
        limit: Döndürülecek maksimum ilaç sayısı

    Returns:
        List[Document]: LLM'e verilecek bağlam belgeleri
        [] (boş liste) eğer hiçbir sonuç bulunamazsa
    """
    if not settings.use_medicine_api:
        return []
    console.print(f"\n[bold cyan]--- Hybrid RAG: API Retrieval ---[/bold cyan]")
    console.print(f"[dim]Sorgu: {query}[/dim]")

    # 1. Entity Extraction
    console.print("\n[yellow]Adim 1: Ilac adi cikartiliyor...[/yellow]")
    medicine_name = extract_medicine_name(query)

    if not medicine_name:
        console.print("[red]Ilac adi tespit edilemedi. API sorgusu atiliyor.[/red]")
        return []

    console.print(f"[green]Tespit edilen ilac: '{medicine_name}'[/green]")

    # 2. API Health Check
    console.print(f"\n[yellow]Adim 2: API saglik kontrolu ({MEDICINE_API_URL})...[/yellow]")
    if not check_api_health():
        console.print(
            f"[red]API erisilemiyor: {MEDICINE_API_URL}[/red]\n"
            f"[dim]Not: Turkish Medicine API'yi baslatmayi unutmayin:[/dim]\n"
            f"[dim]  cd turkish-medicine-api && npm start[/dim]"
        )
        return []

    console.print("[green]API saglıklı ve erisebilir![/green]")

    # 3. API'ye Sorgulama
    console.print(f"\n[yellow]Adim 3: API'ye '{medicine_name}' icin sorgu atiliyor...[/yellow]")
    try:
        results = search_medicine(medicine_name, limit=limit)
    except ConnectionError as e:
        console.print(f"[red]{e}[/red]")
        return []
    except ValueError as e:
        console.print(f"[red]API Hatasi: {e}[/red]")
        return []

    if not results:
        console.print(f"[yellow]'{medicine_name}' icin API'de sonuc bulunamadi.[/yellow]")
        return []

    console.print(f"[green]{len(results)} ilac kaydi bulundu.[/green]")

    # 4. Document Oluşturma
    console.print(f"\n[yellow]Adim 4: JSON → LangChain Document donusumu...[/yellow]")
    documents = build_documents_from_results(results, query=query)
    console.print(f"[green]{len(documents)} Document olusturuldu.[/green]")

    return documents


# ─────────────────────────────────────────────
# 5. Yardımcı: Sonuçları Görselleştir
# ─────────────────────────────────────────────

def display_api_documents(documents: List[Document]) -> None:
    """API'den gelen Document'leri terminalde güzel biçimde gösterir."""
    if not documents:
        console.print("[red]Gösterilecek belge yok.[/red]")
        return

    console.print(f"\n[bold magenta]Bulunan Ilaclar ({len(documents)} kayit):[/bold magenta]\n")

    for i, doc in enumerate(documents, 1):
        m = doc.metadata
        panel_content = (
            f"[cyan]Ilac Adi    :[/cyan] {m.get('medicine_name', '-')}\n"
            f"[cyan]Firma       :[/cyan] {m.get('company', '-')}\n"
            f"[cyan]ATC Kodu    :[/cyan] {m.get('atc_code', '-')}\n"
            f"[cyan]Recete Turu :[/cyan] {m.get('prescription_type', '-')}\n"
            f"[cyan]Durumu      :[/cyan] {m.get('status', '-')}\n"
            f"[cyan]Kaynak      :[/cyan] {m.get('source', '-')}\n"
            f"[dim]page_content uzunlugu: {len(doc.page_content)} karakter[/dim]"
        )
        console.print(Panel(
            panel_content,
            title=f"[bold green]Kayit #{i}[/bold green]",
            border_style="green"
        ))
