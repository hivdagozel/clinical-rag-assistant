# Medical RAG Assistant

Resmî TİTCK KT/KÜB PDF belgelerine dayalı, kaynaklarını backend metadata'sından gösteren Türkçe ilaç bilgi asistanı.

## Güvenlik sınırı

- Klinik kullanım ve güvenlik cevapları yalnız doğrulanmış KT/KÜB PDF chunk'larından üretilir.
- Medicine API yalnız firma, barkod, ruhsat, ATC ve reçete türü metadata'sı sağlar.
- Klinik belge bulunmazsa LLM çağrılmaz.
- Bu uygulama teşhis veya tedavi önerisi yerine geçmez.

## Mimari

```text
Tarayıcı -> FastAPI /api/ask -> HybridRetriever
                              |- FAISS: KT/KÜB PDF chunk'ları
                              `- İsteğe bağlı Medicine API: ürün metadata'sı
          <- cevap + programatik kaynak listesi <- Gemini
```

Sorgular önce deterministik `QueryRouter` katmanından geçer:

- `medicine_clinical`: KT/KÜB tabanlı ilaç RAG
- `medicine_metadata`: ürün/firma/ruhsat metadata akışı
- `symptom`: ilaç önermeyen genel triage
- `emergency`: kısa ve doğrudan acil sağlık yönlendirmesi
- `unsupported`: kapsam dışı sorular

Belirti ve acil durum sorguları ilaç FAISS indeksini veya LLM'i çağırmaz.

Merkezi yollar ve ayarlar `src/config.py` içindedir. Uygulama çalışma dizinine değil dosya konumuna göre proje kökünü hesaplar.

## Klasörler

```text
data/accepted_pdfs/kt/    doğrulanmış kullanma talimatları
data/accepted_pdfs/kub/   doğrulanmış kısa ürün bilgileri
data/quarantine/          scraper doğrulaması bekleyenler
data/rejected_pdfs/       reddedilen, silinmeyen belgeler
data/metadata/            belge metadata'sı
data/vectorstore/         FAISS + manifest (üretilen veri)
src/                      uygulama ve RAG modülleri
static/                   güvenli web arayüzü
tests/                    offline, online ve integration testleri
```

## Kurulum (Windows)

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

`.env` içinde en azından geçerli bir Gemini anahtarı tanımlayın:

```dotenv
LLM_PROVIDER=gemini
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-3.5-flash

EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
USE_MEDICINE_API=false
```

Embedding sağlayıcısı değiştiğinde manifest uyumsuz sayılır ve indeks güvenli biçimde yeniden oluşturulur. Sağlayıcılar: `huggingface`, `gemini`, `openai`; `fake` yalnız test modunda kabul edilir. Bir sağlayıcı hata verirse diğerine sessiz geçiş yapılmaz.

## PDF ekleme ve indeks

Doğrulanmış KT dosyalarını `data/accepted_pdfs/kt`, KÜB dosyalarını `data/accepted_pdfs/kub` altına koyun. Uygulama başlangıcında PDF, metadata, chunk ayarı veya embedding kimliği değişmişse yeni indeks geçici dizinde kurulur, doğrulanır ve sonra aktif edilir. Önceki indeks `.backup` dizininde korunur.

Kontrollü TİTCK KT pilotu:

```powershell
.\venv\Scripts\python.exe scripts\collect_titck_documents.py --type kt --limit 20 --resume --delay 1.5
```

Collector yalnız resmî HTTPS TİTCK adreslerini kabul eder; PDF imzası, metin, ürün adı, belge başlıkları ve SHA-256 tekrarlarını doğrular. CAPTCHA, giriş ekranı, şema değişikliği, art arda HTTP hataları, düşük kabul oranı, yetersiz disk veya yazılamayan checkpoint durumlarında güvenli biçimde durur.

## Mevcut örnek veri

- 9 doğrulanmış KT PDF
- 89 sayfa
- 249 FAISS chunk
- 9 benzersiz ürün
- 384 boyutlu normalize çok dilli embedding

## Çalıştırma

```powershell
.\venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Arayüz: `http://127.0.0.1:8000`

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/status
Invoke-RestMethod http://127.0.0.1:8000/api/ask -Method Post -ContentType application/json -Body '{"question":"Parol nasıl kullanılır?"}'
```

## Medicine API

Ayrı Turkish Medicine API servisi isteğe bağlıdır. `USE_MEDICINE_API=false` iken hiçbir çağrı yapılmaz ve klinik PDF soruları çalışmaya devam eder. Açıkken sağlık kontrolü `/health` üzerinden yapılır.

## Testler

```powershell
.\venv\Scripts\python.exe -m pytest -m offline -v
.\venv\Scripts\python.exe -m pytest -m integration -v
.\venv\Scripts\python.exe -m pytest -m online -v
```

Offline testler benzersiz geçici `TEST_VECTORSTORE_DIR` ve deterministik fake embedding kullanır; gerçek `data/vectorstore` dizinine dokunmaz.

## Sorun giderme

- `missing_api_key`: `.env` içindeki `GEMINI_API_KEY` değerini kontrol edin.
- `vectorstore_error`: sunucu logundaki embedding/manifest/rebuild hatasını inceleyin.
- Model değişti: indeks manifest nedeniyle otomatik yeniden kurulur.
- Medicine API offline: klinik PDF akışı devam eder; firma/barkod soruları doğrulanamaz.
- Windows Torch DLL/yol sorunu: Python 3.12 uyumlu CPU Torch kurun ve sanal ortam yolunu kısa tutun.

Status endpoint'i LLM, embedding, FAISS ve Medicine API durumlarını ayrı ayrı raporlar. Kullanıcıya stack trace veya gizli anahtar döndürülmez.
