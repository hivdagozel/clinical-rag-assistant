"""Rule-based, non-diagnostic symptom and emergency responses."""
from __future__ import annotations

from src.query_analysis import normalize_text


EMERGENCY_RESPONSE = (
    "Bu belirtiler acil değerlendirme gerektirebilir. Türkiye'deyseniz 112'yi arayın "
    "veya en yakın acil servise başvurun. Kendiniz araç kullanmayın; mümkünse yanınızda "
    "biri olsun. Buradan tanı koymak veya ilaç önermek güvenli değildir."
)

HEADACHE_RESPONSE = """Baş ağrısının birçok nedeni olabilir. Ne zamandır sürdüğü, aniden başlayıp başlamadığı, şiddeti ve eşlik eden belirtiler önemlidir.

Şunlardan biri varsa acil değerlendirme gerekir:
- Hayatınızdaki en şiddetli veya aniden başlayan baş ağrısı
- Konuşma bozukluğu, yüzde kayma ya da kol/bacakta güçsüzlük
- Bayılma, bilinç bulanıklığı veya nöbet
- Ateş ve ense sertliği
- Kafa travmasından sonra başlayan ağrı

Bunlar yoksa dinlenme, yeterli sıvı alma ve belirtileri takip etme yardımcı olabilir.

Baş ağrısında yetişkinler için reçetesiz kullanılan yaygın seçenekler:
- Parasetamol içeren ilaçlar ağrıyı azaltmak için kullanılabilir. Karaciğer hastalığınız varsa, düzenli alkol kullanıyorsanız veya başka bir ilaç da parasetamol içeriyorsa kullanmadan önce eczacıya ya da doktora danışın.
- İbuprofen gibi iltihap giderici ağrı kesiciler bazı baş ağrılarında yardımcı olabilir. Mide ülseri/kanama öyküsü, böbrek hastalığı, kan sulandırıcı kullanımı, gebelik veya bu ilaçlara bağlı astım/alerji varsa uygun olmayabilir.

Size uygun seçenek; yaşınıza, hastalıklarınıza, kullandığınız diğer ilaçlara ve gebelik durumuna göre değişir. Aynı etken maddeyi içeren ürünleri birlikte kullanmayın, kutudaki kullanma talimatına ve belirtilen doza uyun; emin değilseniz eczacınıza danışın. Ağrı tekrarlıyor, şiddetleniyor veya uzun sürüyorsa bir sağlık profesyoneline başvurun.

Daha iyi yönlendirme için: Ağrı ne zamandır var, aniden mi başladı, 0–10 arasında şiddeti kaç ve başka bir belirti eşlik ediyor mu?"""

ABDOMINAL_RESPONSE = """Karın ağrısının farklı nedenleri olabilir; buradan tanı koymak veya ilaç önermek güvenli değildir.

Şiddetli ya da giderek artan ağrı, karında sertlik, bayılma, kanlı kusma/dışkı, sürekli kusma, yüksek ateş veya gebelik ihtimali varsa acil değerlendirme alın. Sağ alt karında belirginleşen ağrı da gecikmeden değerlendirilmelidir.

Daha iyi yönlendirme için: Ağrı tam olarak nerede, ne zamandır var, 0–10 arasında şiddeti kaç? Kusma, ateş, ishal, kabızlık, kanama, idrar yakınması veya gebelik ihtimali var mı?"""

GENERIC_RESPONSE = """Bu belirti farklı nedenlerle ortaya çıkabilir; buradan tanı koymak veya ilaç/doz önermek güvenli değildir.

Belirti aniden başladıysa, çok şiddetliyse, hızla kötüleşiyorsa; nefes darlığı, bayılma, bilinç değişikliği veya ciddi kanama eşlik ediyorsa acil sağlık hizmetine başvurun. Bunlar yoksa belirtileri izleyin ve sürmesi ya da tekrarlaması halinde bir sağlık profesyoneline danışın.

Daha iyi yönlendirme için belirtinin ne zamandır sürdüğünü, şiddetini ve eşlik eden diğer belirtileri yazın."""


def symptom_triage_response(query: str) -> tuple[str, list[str]]:
    value = normalize_text(query)
    if "bas" in value and ("agri" in value or "don" in value):
        return HEADACHE_RESPONSE, ["Ağrı ne zamandır var?", "Ağrı ne kadar şiddetli?", "Başka bir belirti eşlik ediyor mu?"]
    if "karin" in value or "karnim" in value:
        return ABDOMINAL_RESPONSE, ["Ağrı tam olarak nerede?", "Ne zamandır sürüyor?", "Kusma, ateş veya kanama var mı?"]
    return GENERIC_RESPONSE, ["Belirti ne zamandır var?", "Ne kadar şiddetli?", "Başka bir belirti eşlik ediyor mu?"]
