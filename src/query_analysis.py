"""Conservative Turkish product extraction, normalization and intent classification."""
from __future__ import annotations
import re
import unicodedata
import json
from functools import lru_cache
from enum import Enum

from src.config import METADATA_FILE

def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.casefold().replace("ı", "i")
    return re.sub(r"[^a-z0-9]+", " ", value).strip()

class QueryIntent(str, Enum):
    CLINICAL_USAGE = "clinical_usage"
    CLINICAL_SAFETY = "clinical_safety"
    PRODUCT_METADATA = "product_metadata"
    GENERAL_DOCUMENT = "general_document"

def classify_intent(query: str) -> QueryIntent:
    value = normalize_text(query)
    groups = (
        (QueryIntent.PRODUCT_METADATA, ("firma", "barkod", "ruhsat", "uretici", "atc", "recete turu", "kayit")),
        (QueryIntent.CLINICAL_SAFETY, ("yan etki", "gebelik", "emzirme", "kontrendikasyon", "etkilesim", "uyari", "saklan")),
        (QueryIntent.CLINICAL_USAGE, ("nasil kullan", "doz", "kac kez", "ac mi", "tok mu", "kullanim sekli", "gunde")),
    )
    for intent, phrases in groups:
        if any(phrase in value for phrase in phrases):
            return intent
    return QueryIntent.GENERAL_DOCUMENT

KNOWN_BRANDS = ("parol plus", "parol hot", "parol", "parasetamol", "aspirin", "majezik", "dolven", "aferin", "augmentin", "coraspin", "cipro", "lansor")
PRODUCT_ALIASES = {"parasetamol": {"parol"}}
GENERIC_PRODUCT_TOKENS = {
    "mg", "ml", "iv", "icin", "icinde", "iceren", "cozelti", "cozeltisi", "tablet", "kapsul",
    "surup", "oral", "film", "infuzyon", "enjeksiyonluk", "dekstroz", "sodyum",
    "klorur", "ringer", "biosel", "sudaki", "hipertonik", "ampul", "flakon",
    "damla", "damlasi", "yumusak", "sert", "kapli", "goz", "konsantre",
}

@lru_cache(maxsize=1)
def metadata_product_aliases() -> set[str]:
    """Return distinctive product tokens from accepted document metadata."""
    try:
        payload = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    aliases: set[str] = set()
    for item in payload.values() if isinstance(payload, dict) else ():
        if not isinstance(item, dict) or item.get("status") != "success":
            continue
        for token in normalize_text(item.get("drug_name", "")).split():
            if len(token) >= 4 and not token.isdigit() and token not in GENERIC_PRODUCT_TOKENS:
                aliases.add(token)
        ingredients = item.get("active_ingredients", [])
        if isinstance(ingredients, str):
            ingredients = [ingredients]
        for ingredient in ingredients if isinstance(ingredients, list) else ():
            phrase = normalize_text(ingredient)
            if phrase:
                aliases.add(phrase)
    return aliases


@lru_cache(maxsize=1)
def metadata_active_ingredient_aliases() -> set[str]:
    try:
        payload = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    aliases: set[str] = set()
    for item in payload.values() if isinstance(payload, dict) else ():
        if not isinstance(item, dict) or item.get("status") != "success":
            continue
        ingredients = item.get("active_ingredients", [])
        if isinstance(ingredients, str):
            ingredients = [ingredients]
        aliases.update(normalize_text(value) for value in ingredients if str(value).strip())
    return aliases


def metadata_matching_products(query_product: str | None) -> list[str]:
    """List accepted product labels matching a normalized query product."""
    if not query_product:
        return []
    try:
        payload = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    query_tokens = normalize_text(query_product).split()
    query_variants = {"plus", "forte", "hot", "pediatrik"}.intersection(query_tokens)
    root_names = PRODUCT_ALIASES.get(" ".join(query_tokens), {query_tokens[0] if query_tokens else ""})
    names: set[str] = set()
    for item in payload.values() if isinstance(payload, dict) else ():
        if not isinstance(item, dict) or item.get("status") != "success":
            continue
        name = str(item.get("drug_name", "")).strip()
        document_tokens = set(normalize_text(name).split())
        ingredients = item.get("active_ingredients", [])
        if isinstance(ingredients, str):
            ingredients = [ingredients]
        ingredient_names = {normalize_text(value) for value in ingredients if str(value).strip()}
        ingredient_match = any(
            normalize_text(query_product) == value
            or normalize_text(query_product) in value.split()
            for value in ingredient_names
        )
        matches = product_matches(query_product, name) if query_variants else (
            bool(root_names.intersection(document_tokens)) or ingredient_match
        )
        if matches:
            names.add(name)
    names.discard("")
    return sorted(names)

def symptom_terms(query: str) -> tuple[str, ...]:
    value = normalize_text(query)
    if "bas" in value and "agri" in value:
        return ("bas agri", "bas agrisi")
    if "ates" in value:
        return ("ates",)
    if "agri" in value:
        return ("agri",)
    return ()

def is_symptom_query(query: str) -> bool:
    value = normalize_text(query)
    recommendation_words = ("hangi ilac", "ne kullan", "ne al", "iyi gelir", "oner")
    return bool(symptom_terms(query)) and any(phrase in value for phrase in recommendation_words)

def extract_product_name(query: str) -> str | None:
    normalized = normalize_text(query)
    for brand in KNOWN_BRANDS:
        if re.search(rf"(?<![a-z0-9]){re.escape(brand)}(?![a-z0-9])", normalized):
            return brand
    for alias in sorted(metadata_product_aliases(), key=len, reverse=True):
        if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", normalized):
            variants = [variant for variant in ("plus", "forte", "hot", "pediatrik") if variant in normalized.split()]
            return " ".join([alias, *variants])
    if is_symptom_query(query):
        return None
    quoted = re.findall(r'["\']([^"\']+)["\']', query)
    if quoted:
        return normalize_text(quoted[0]) or None
    # Serbest metindeki ilk kelimeyi ilaç adı sayma: belirti ve kapsam dışı
    # sorguların yanlışlıkla ilaç RAG akışına girmesine neden olur.
    return None

def product_matches(query_product: str | None, metadata_name: str) -> bool:
    if not query_product:
        return False
    query_tokens = normalize_text(query_product).split()
    document_tokens = normalize_text(metadata_name).split()
    if not query_tokens or not document_tokens:
        return False
    accepted_names = PRODUCT_ALIASES.get(" ".join(query_tokens), {query_tokens[0]})
    if not accepted_names.intersection(document_tokens):
        return False
    variants = {"plus", "forte", "hot", "pediatrik"}
    return variants.intersection(query_tokens) == variants.intersection(document_tokens)


def product_matches_metadata(query_product: str | None, metadata: dict) -> bool:
    """Match a brand/canonical variant or a collected active ingredient."""
    if not query_product:
        return False
    ingredients = metadata.get("active_ingredients", [])
    if isinstance(ingredients, str):
        ingredients = [ingredients]
    query = normalize_text(query_product)
    for ingredient in ingredients if isinstance(ingredients, list) else ():
        value = normalize_text(ingredient)
        if query == value or query in value.split():
            return True
    if query in metadata_active_ingredient_aliases():
        return False
    name = str(
        metadata.get("normalized_drug_name")
        or metadata.get("drug_name")
        or metadata.get("medicine_name")
        or ""
    )
    return product_matches(query_product, name)

def document_priority(intent: QueryIntent, document_type: str, source_type: str) -> int:
    doc_type = document_type.upper()
    if intent == QueryIntent.PRODUCT_METADATA:
        return 0 if source_type == "api" else 2
    if intent == QueryIntent.CLINICAL_USAGE:
        return 0 if doc_type == "KT" else 1 if doc_type in {"KÜB", "KUB"} else 3
    if intent == QueryIntent.CLINICAL_SAFETY:
        return 0 if doc_type in {"KT", "KÜB", "KUB"} else 3
    return 0 if doc_type == "KT" else 1 if doc_type in {"KÜB", "KUB"} else 3
