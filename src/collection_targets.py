"""Configuration-driven, deterministic medicine collection targeting."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


def normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value)).casefold().replace("ı", "i")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def first_value(record: dict[str, Any], names: Iterable[str]) -> str:
    for name in names:
        value = record.get(name)
        if value not in (None, ""):
            if isinstance(value, list):
                return " + ".join(map(str, value))
            return str(value)
    return ""


@dataclass(frozen=True)
class TargetMatch:
    categories: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    active_ingredients: tuple[str, ...] = ()
    priority: int = 999
    combination_product: bool = False

    @property
    def selected(self) -> bool:
        return bool(self.categories)


class TargetSelector:
    def __init__(self, config_path: Path, categories: Iterable[str] = ()):
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        configured = payload.get("categories", {})
        if not isinstance(configured, dict):
            raise ValueError("categories bir YAML nesnesi olmalıdır")
        requested = {normalize(item) for item in categories if str(item).strip()}
        unknown = requested.difference(configured)
        if unknown:
            raise ValueError(f"Bilinmeyen kategoriler: {', '.join(sorted(unknown))}")
        self.categories = {
            name: value for name, value in configured.items()
            if isinstance(value, dict) and value.get("enabled", True)
            and (not requested or name in requested)
        }

    @staticmethod
    def record_fields(record: dict[str, Any]) -> tuple[str, str, str]:
        product = first_value(record, ("name", "productName", "product_name", "medicineName"))
        ingredients = first_value(record, (
            "activeIngredient", "activeIngredients", "active_ingredient",
            "activeSubstance", "activeSubstances", "ingredient", "element",
        ))
        atc = first_value(record, ("atcCode", "atc_code", "atc", "ATC"))
        return product, ingredients, atc

    def match(self, record: dict[str, Any]) -> TargetMatch:
        product, ingredients, atc = self.record_fields(record)
        product_norm, ingredients_norm, atc_norm = map(normalize, (product, ingredients, atc))
        searchable = f"{ingredients_norm} {product_norm}".strip()
        matched_categories: list[str] = []
        reasons: list[str] = []
        matched_ingredients: list[str] = []
        priorities: list[int] = []
        for name, definition in self.categories.items():
            excluded = [normalize(item) for item in definition.get("exclude_terms", [])]
            if any(term and term in searchable for term in excluded):
                continue
            category_reasons: list[str] = []
            for prefix in definition.get("atc_prefixes", []):
                if atc_norm.replace(" ", "").startswith(normalize(prefix).replace(" ", "")):
                    category_reasons.append(f"atc_prefix:{str(prefix).upper()}")
            for ingredient in definition.get("active_ingredients", []):
                ingredient_norm = normalize(ingredient)
                # Prefer the official ingredient field; product-name matching is
                # an explicit last-resort fallback for generic-named products.
                if ingredient_norm and (
                    ingredient_norm in ingredients_norm
                    or (not ingredients_norm and ingredient_norm in product_norm)
                ):
                    matched_ingredients.append(str(ingredient))
                    category_reasons.append(f"active_ingredient:{ingredient_norm}")
            if category_reasons:
                matched_categories.append(name)
                reasons.extend(category_reasons)
                priorities.append(int(definition.get("priority", 999)))
        unique_ingredients = tuple(dict.fromkeys(matched_ingredients))
        combination = len(unique_ingredients) > 1 or bool(re.search(r"\+|/|\bve\b", ingredients, flags=re.I))
        return TargetMatch(
            tuple(matched_categories), tuple(dict.fromkeys(reasons)), unique_ingredients,
            min(priorities, default=999), combination,
        )


STRENGTH_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|g|iu|ünite|%)(?:\s*/\s*\d+(?:[.,]\d+)?\s*ml)?", re.I)
FORMS = (
    "efervesan tablet", "film kaplı tablet", "uzatılmış salımlı tablet",
    "süspansiyon", "şurup", "enjeksiyonluk çözelti", "infüzyon", "kapsül",
    "tablet", "damla", "sprey", "krem", "merhem", "jel", "flakon", "ampul",
)
RELEASE_TYPES = ("sr", "mr", "xr", "retard", "uzatılmış salımlı", "kontrollü salım")


def canonical_product_fields(product_name: str, ingredients: Iterable[str]) -> dict[str, Any]:
    value = normalize(product_name)
    strength_match = STRENGTH_RE.search(product_name)
    dosage_form = next((form for form in FORMS if normalize(form) in value), "")
    release_type = next((item for item in RELEASE_TYPES if normalize(item) in value), "immediate")
    brand = value.split()[0] if value else ""
    ingredient_values = list(dict.fromkeys(str(item) for item in ingredients if str(item).strip()))
    key_parts = [brand, normalize(strength_match.group(0)) if strength_match else "", normalize(dosage_form), normalize(release_type)]
    return {
        "normalized_brand": brand,
        "active_ingredients": ingredient_values,
        "strength": strength_match.group(0) if strength_match else "",
        "dosage_form": dosage_form,
        "release_type": release_type,
        "canonical_product_key": "|".join(key_parts),
    }
