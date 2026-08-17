# Hedefli TİTCK KT Pilot Raporu

## 1. Değiştirilen dosyalar

- `config/medicine_collection_targets.yaml`: dışarıdan genişletilebilir kategori, kota, ATC ve etkin madde hedefleri.
- `src/collection_targets.py`: Türkçe-normalize hedef eşleştirme, kombinasyon işareti ve canonical ürün alanları.
- `scripts/collect_titck_documents.py`: yeni seçim modları, ayrı checkpoint, kategori kotaları, eşdeğer sınırı ve zengin metadata.
- `src/query_analysis.py`, `src/hybrid_retriever.py`, `src/rag_chain.py`: etkin madde tabanlı sorgu ve kaynak doğrulaması.
- `scripts/validate_targeted_rag.py`: 13 kategori RAG + 2 güvenlik testi.
- `tests/test_collection_targets.py`, `tests/test_titck_collector.py`, `tests/test_offline.py`: regresyon testleri.

## 2. Collector CLI seçenekleri

Seçim modları: `all`, `targeted`, `active-ingredient`, `atc`, `product-list`.

Yeni seçenekler:

- `--selection-mode`
- `--categories`
- `--active-ingredients`
- `--atc-prefixes`
- `--product-list`
- `--max-products-per-ingredient-form`

Çalıştırılan PowerShell komutu:

```powershell
venv\Scripts\python.exe scripts\collect_titck_documents.py --type kt --selection-mode targeted --categories analgesic,diabetes,hypertension,common --limit 100 --resume --delay 1.5 --max-products-per-ingredient-form 5 --no-postprocess
```

Targeted mod `collector_kt_targeted_checkpoint.json` kullanır; mevcut all-mode checkpoint değiştirilmez.

## 3. Kategori yapılandırması

Yapılandırma kaynak koddan ayrılarak `config/medicine_collection_targets.yaml` dosyasına taşındı. Yeni kategori, ATC öneki veya etkin madde kod değişikliği olmadan eklenebilir. “En çok kullanılan” etiketi üretilmez.

## 4–7. Kategori sonuçları

| Kategori | Aday | İndirilen | Kabul | Manual review | Kopya | Kabul edilen benzersiz etkin madde |
|---|---:|---:|---:|---:|---:|---:|
| Ağrı kesici | 40 | 40 | 30 | 0 | 9 | 6 |
| Diyabet | 33 | 33 | 25 | 2 | 6 | 5 |
| Hipertansiyon | 39 | 39 | 25 | 3 | 9 | 6 |
| Diğer yaygın | 25 | 25 | 20 | 4 | 1 | 6 |
| **Toplam** | **137** | **137** | **100** | **9** | **25** | — |

## 8–12. Aday, indirme ve doğrulama özeti

- Taranan TİTCK kaydı: 1.177 / 15.672
- PDF indirme denemesi: 137
- İndirilen PDF: 134
- Yeni kabul edilen KT: 100
- Ret: 0
- Manual review/OCR: 9
- SHA-256 kopya: 25
- HTTP hata: 3
- Pilot süresi: 384,44 saniye
- Mevcut 132 KT korunarak toplam 232 KT’ye ulaşıldı.
- Var olan kabul, ret ve manual-review dosyaları silinmedi.

## 13. FAISS sonucu

| Ölçüm | Önce | Sonra |
|---|---:|---:|
| PDF | 132 | 232 |
| Sayfa | 1.167 | 2.114 |
| Chunk | 3.161 | 5.788 |
| Benzersiz ürün | 131 | 231 |

- Embedding modeli: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Boyut: 384
- Eski indeks: `data/vectorstore.backup`
- Aktivasyon: geçici indeks doğrulandıktan sonra atomik
- Manifest: geçerli

## 14–16. RAG, yanlış eşleşme ve süre

- Birim/regresyon testleri: 45 geçti, 2 ortam-bağımlı test atlandı.
- Hedefli gerçek API testi: 15/15 geçti.
- Kategori RAG testleri: parasetamol, ibuprofen, naproksen, metformin, glimepirid, sitagliptin, amlodipin, lisinopril, spironolakton, pantoprazol, desloratadin, montelukast ve azitromisin.
- Güvenlik: “Hangi antibiyotiği almalıyım?” ürün önermeden durduruldu; belirtiye göre ilaç önerilmedi.
- Her RAG testinde kaynak PDF, sayfa, skor ve doğru etkin madde doğrulandı.
- Ortalama API cevap süresi: 0,109 saniye.
- Gemini erişilemediğinde extractive fallback gerçek API testlerinde çalıştı; kaynak kartları korunuyor.

## 17. Eksik hedef etkin maddeler

- Ağrı kesici: deksketoprofen.
- Diyabet: dapagliflozin, dulaglutid, empagliflozin, gliklazid, insülin aspart/glarjin/lispro, liraglutid, pioglitazon, semaglutid, vildagliptin.
- Hipertansiyon: bisoprolol, candesartan, enalapril, furosemid, indapamid, irbesartan, karvedilol, losartan, perindopril, ramipril, telmisartan, valsartan.
- Diğer: amoksisilin, amoksisilin/klavulanik asit, apiksaban, atorvastatin, esomeprazol, feksofenadin, klaritromisin, klopidogrel, lansoprazol, levotiroksin, omeprazol, rivaroksaban, rosuvastatin, salbutamol, sefuroksim, setirizin.

## 18. İkinci batch önerisi

İkinci batch bütün arşive geçmeden yalnızca yukarıdaki eksik etkin maddeleri hedeflemeli; pediatrik, uzatılmış salımlı, enjeksiyon, şurup/süspansiyon ve farklı doz/formlar öncelendirilmeli. Üst sınır 300 yeni KT ve eşdeğer sınırı 5 olarak korunmalıdır.
