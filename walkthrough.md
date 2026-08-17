# Uygulama düzeltme walkthrough'u

## Yapılan değişiklikler

1. `src/config.py` ile bütün veri, PDF, metadata, statik ve FAISS yolları tek merkezde toplandı.
2. Loader yalnız doğrulanmış `accepted_pdfs/kt` ve `accepted_pdfs/kub` dizinlerini okur; KÜB dizini boş olabilir.
3. Manifest PDF içerikleri, metadata, embedding sağlayıcı/model/boyut, chunk ayarları ve şema sürümlerini izler.
4. FAISS yeni indeksi geçici dizinde oluşturur, yeniden yükleyip doğrular, eski indeksi yedekleyerek aktive eder.
5. `TEST_MODE` gerçek, deterministik ve ağsız embedding kullanır; test indeks yolu geçicidir.
6. Embedding sağlayıcı seçimi deterministiktir; sessiz sağlayıcı fallback'i kaldırıldı.
7. Medicine API opsiyonel oldu ve health/status aynı `/health` kontrolünü kullanır.
8. Soru niyetleri clinical_usage, clinical_safety, product_metadata ve general_document olarak sınıflandırılır.
9. Parol, Parol Plus ve Parol Hot ayrımı normalize metadata üzerinden yapılır.
10. Hybrid retrieval API sonucundan bağımsız olarak PDF aramasını çalıştırır; keyword fallback her zaman `(Document, score)` döndürür.
11. Belge yokluğu ve LLM üretim hatası farklı durumlar/mesajlar olarak yönetilir.
12. Kaynak listesi LLM tarafından değil doğrudan retrieved Document metadata'sından oluşturulur.
13. FastAPI import-time başlatma kaldırıldı; lifespan, degraded mode ve thread offload eklendi.
14. Frontend gerçek status bilgisini gösterir; çift gönderimi engeller, Shift+Enter destekler ve kaynak DOM'unu `textContent` ile üretir.
15. Scraper TLS doğrulamasını kapatmaz; quarantine/accepted/rejected akışını kullanır.
16. Requirements, `.env.example`, `.gitignore`, README ve test marker'ları güncellendi.

## Doğrulama

- Offline testler: deterministik fake embedding ve geçici FAISS ile çalışır.
- Integration test: ana sayfa, statik JS, status ve ask endpoint'lerini FastAPI TestClient ile doğrular.
- `pytest -m "offline or integration"`: 15 test geçti.
- `pytest -m online`: Medicine API kapalı ve sandbox DNS erişimi olmadığı için 2 test skip edildi.
- Gerçek HuggingFace modeli çevrimdışı cache'den yüklendi; embedding boyutu 384 olarak doğrulandı.
- Gerçek indeks 6 PDF / 57 sayfa / 174 chunk / 6 benzersiz ilaç ile yeniden kuruldu.
- Manifest PDF ve metadata hash'leriyle birlikte geçerli olarak doğrulandı.
- `gemini-2.5-flash` hesabın API çağrısında 404 verdi; resmî model listesi ve minimal üretim çağrısıyla `gemini-3.5-flash` doğrulandı.
- Gerçek `POST /api/ask` çağrısı “Parol nasıl kullanılır?” için HTTP 200 döndürdü; 5 KT chunk'ı ve sayfa 1, 4, 5, 6 kullanıldı, PAROL PLUS/HOT kullanılmadı.
- In-app browser ile localhost arayüzü açıldı; sistem göstergeleri (174 chunk, 6 ilaç, 6 PDF), soru gönderimi, ağ/model hatasının kullanıcıya gösterilmesi, formun yeniden etkinleşmesi ve temiz tarayıcı konsolu doğrulandı. Bu çalışma ortamının dış ağ/DNS kısıtı nedeniyle canlı Gemini yanıtı tarayıcı içinde üretilemedi; gerçek Gemini yanıtı ve PAROL kaynakları ayrıca HTTP uçtan uca testiyle doğrulandı.
