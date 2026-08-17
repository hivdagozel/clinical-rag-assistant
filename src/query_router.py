"""Deterministic Turkish query routing for medical safety boundaries."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.query_analysis import classify_intent, extract_product_name, normalize_text


class QueryRoute(str, Enum):
    MEDICINE_CLINICAL = "medicine_clinical"
    MEDICINE_METADATA = "medicine_metadata"
    SYMPTOM = "symptom"
    EMERGENCY = "emergency"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class RouteResult:
    route: QueryRoute
    product: str | None
    intent: str
    reason: str


class QueryRouter:
    EMERGENCY_PATTERNS = (
        ("gogus", "nefes"), ("nefes alam",), ("kan kus",), ("bayil",),
        ("yuz", "uyus"), ("yuz", "kayma"), ("konusma", "bozuk"),
        ("aniden", "siddetli", "bas", "agri"), ("karin", "siddetli", "baygin"),
    )
    SYMPTOM_WORDS = (
        "basim agri", "bas agrisi", "karnim agri", "karin agri", "midem bulan",
        "atesim", "ates var", "basim don", "bas don", "bogazim agri", "oksur",
        "ishal", "kusma", "nefes darligi", "halsiz",
    )
    MEDICINE_WORDS = ("ilac", "tablet", "surup", "kapsul", "doz", "kullan", "yan etki", "saklan")
    METADATA_WORDS = ("firma", "barkod", "ruhsat", "uretici", "atc", "recete turu")

    def classify(self, query: str) -> RouteResult:
        value = normalize_text(query)
        product = extract_product_name(query)
        intent = classify_intent(query).value

        if any(all(token in value for token in pattern) for pattern in self.EMERGENCY_PATTERNS):
            return RouteResult(QueryRoute.EMERGENCY, product, intent, "emergency_pattern")
        if any(phrase in value for phrase in self.SYMPTOM_WORDS):
            return RouteResult(QueryRoute.SYMPTOM, None, intent, "symptom_pattern")
        if any(word in value for word in self.METADATA_WORDS):
            return RouteResult(QueryRoute.MEDICINE_METADATA, product, intent, "metadata_pattern")
        if product or any(word in value for word in self.MEDICINE_WORDS):
            return RouteResult(QueryRoute.MEDICINE_CLINICAL, product, intent, "medicine_pattern")
        return RouteResult(QueryRoute.UNSUPPORTED, None, intent, "out_of_scope")
