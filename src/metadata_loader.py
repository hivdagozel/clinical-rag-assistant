import json
import logging
from pathlib import Path
from src.config import METADATA_FILE

logger = logging.getLogger(__name__)

DOCUMENTS_JSON_PATH = METADATA_FILE

class MetadataLoader:
    """
    TİTCK scripti tarafindan toplanan metadata JSON dosyasini okur
    ve PDF dosya ismine veya hash'e gore metadata dondurur.
    """
    def __init__(self, json_path: str = None):
        self.json_path = Path(json_path) if json_path else DOCUMENTS_JSON_PATH
        self.metadata_index = {} # file_name -> metadata mapping
        self._load_metadata()

    def _load_metadata(self):
        if not self.json_path.exists():
            logger.warning(f"Metadata dosyasi bulunamadi: {self.json_path}")
            return

        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return
                data = json.loads(content)

            # JSON formatimiz hash bazli: {"hash": {"file_name": "...", "drug_name": "..."}}
            records = data.values() if isinstance(data, dict) else data
            for record in records:
                file_name = record.get("file_name")
                if file_name:
                    normalized = dict(record)
                    normalized["document_type"] = str(normalized.get("document_type", "")).upper()
                    normalized["normalized_drug_name"] = normalize_drug_name(normalized.get("drug_name", ""))
                    self.metadata_index[file_name] = normalized

            logger.info(f"{len(self.metadata_index)} adet metadata kaydi yüklendi.")
        except Exception as e:
            logger.error(f"Metadata yukleme hatasi: {e}")

    def get_metadata_for_file(self, file_path: str) -> dict:
        """
        PDF dosya ismine gore metadata arar ve bulursa dondurur.

        Args:
            file_path: PDF dosyasinin tam yolu veya adi (ornegin: "C:/.../parol_abcd.pdf")

        Returns:
            dict: Bulunan metadata veya bos sozluk.
        """
        file_name = Path(file_path).name
        return self.metadata_index.get(file_name, {})

# Modul seviyesinde kullanilmak uzere global bir instance da tutulabilir
_default_loader = None


def normalize_drug_name(value: str) -> str:
    import re
    import unicodedata
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.casefold().replace("ı", "i")
    return re.sub(r"[^a-z0-9]+", " ", value).strip()

def get_metadata(file_path: str) -> dict:
    """
    Yardimci fonksiyon. Varsayilan MetadataLoader uzerinden islem yapar.
    """
    global _default_loader
    if _default_loader is None:
        _default_loader = MetadataLoader()
    return _default_loader.get_metadata_for_file(file_path)
