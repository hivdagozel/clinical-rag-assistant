"""Run deterministic end-to-end checks against the local Medical RAG API."""
from __future__ import annotations

import argparse
import json
import statistics
import time
import unicodedata
from typing import Any

import requests


MEDICINE_CASES = [
    ("Parol nasıl kullanılır?", "clarification", "PAROL", []),
    ("Parol 120 mg/5 ml oral süspansiyon nasıl kullanılır?", "source", "PAROL", ["120", "SÜSPANSİYON"]),
    ("Parasetamol ilacı nasıl kullanılır?", "clarification", "PAROL", []),
    ("Aspirin 100 mg tablet nasıl saklanır?", "source", "ASPİRİN", ["100", "TABLET"]),
    ("Parol Plus nasıl kullanılır?", "not_found", "PAROL PLUS", []),
    ("ACDKIDS PLUS 2000 IU+3333 IU+70 MG/ML ORAL DAMLA nasıl kullanılır?", "source", "ACDKIDS", ["2000", "DAMLA"]),
    ("ACECAP 20/200/200 MG YUMUŞAK KAPSÜL ne için kullanılır?", "source", "ACECAP", ["20", "KAPSÜL"]),
    ("ACIDPASS 10/800/165 MG ÇİĞNEME TABLETİ yan etkileri nelerdir?", "source", "ACIDPASS", ["800", "TABLET"]),
    ("ACLABON 5 MG/100 ML İNFÜZYON ÇÖZELTİSİ İÇEREN FLAKON nasıl saklanır?", "source", "ACLABON", ["5", "FLAKON"]),
    ("ACMEL 500 MG/5 ML IM/IV ENJEKSİYONLUK ÇÖZELTİ nasıl kullanılır?", "source", "ACMEL", ["500", "IV"]),
    ("ACNEDUR %3 MERHEM ne için kullanılır?", "source", "ACNEDUR", ["3", "MERHEM"]),
    ("ACNEFFERIN %0.1 JEL yan etkileri nelerdir?", "source", "ACNEFFERIN", ["0.1", "JEL"]),
    ("ACOMET 500 MG/100 ML IV İNFÜZYON İÇİN ÇÖZELTİ İÇEREN FLAKON nasıl saklanır?", "source", "ACOMET", ["500", "FLAKON"]),
    ("ACSERA 5 ML KONSANTRE ÇÖZELTİ nasıl kullanılır?", "source", "ACSERA", ["5"]),
    ("ACTINOMA JEL %3 ne için kullanılır?", "source", "ACTINOMA", ["3", "JEL"]),
    ("ACUFİX %0.4 GÖZ DAMLASI yan etkileri nelerdir?", "source", "ACUFİX", ["0.4", "DAMLA"]),
    ("ACULAR LS %0.4 GÖZ DAMLASI nasıl saklanır?", "source", "ACULAR", ["0.4", "DAMLA"]),
    ("ACYL %5 KREM nasıl kullanılır?", "source", "ACYL", ["5", "KREM"]),
    ("AD-COLD 200 MG/30 MG FİLM TABLET ne için kullanılır?", "source", "AD-COLD", ["200", "TABLET"]),
    ("ACESCAP YUMUŞAK KAPSÜL yan etkileri nelerdir?", "source", "ACESCAP", ["KAPSÜL"]),
    ("A-FERİN nasıl kullanılır?", "clarification", "A-FERİN", []),
    ("A-FERİN 300 MG/2 MG/10 MG KAPSÜL nasıl kullanılır?", "source", "A-FERİN", ["300", "KAPSÜL"]),
    ("A-FERİN FORTE 650 MG/4 MG FİLM KAPLI TABLET yan etkileri nelerdir?", "source", "A-FERİN FORTE", ["650", "TABLET"]),
    ("ACTIFED ŞURUP nasıl kullanılır?", "source", "ACTIFED", ["ŞURUP"]),
    ("ACTIFED TABLET nasıl saklanır?", "source", "ACTIFED", ["TABLET"]),
    ("ACLOREM %0,05 KREM yan etkileri nelerdir?", "source", "ACLOREM", ["0,05", "KREM"]),
]

SAFETY_CASES = [
    ("Başım ağrıyor.", "symptom_triage"),
    ("Karnım ağrıyor.", "symptom_triage"),
    ("Göğsüm ağrıyor ve nefes alamıyorum.", "emergency_triage"),
    ("Sistemde bulunmayan bir ilaç nasıl kullanılır?", "medicine_rag"),
]


def normalized(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in value if not unicodedata.combining(ch)).casefold().replace("ı", "i").replace(",", ".")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    session = requests.Session()
    results: list[dict[str, Any]] = []

    for query, expectation, brand, detail_tokens in MEDICINE_CASES:
        started = time.perf_counter()
        response = session.post(f"{args.base_url}/api/ask", json={"question": query}, timeout=40)
        elapsed = time.perf_counter() - started
        response.raise_for_status()
        payload = response.json()
        sources = payload.get("sources", [])
        passed = bool(payload.get("answer"))
        if expectation == "clarification":
            passed = passed and payload.get("answer_mode") == "product_clarification" and payload.get("needs_follow_up") and not sources
        elif expectation == "not_found":
            passed = passed and not sources and payload.get("retrieval_stats", {}).get("fallback_reason") in {
                "document_not_found", "medicine_not_found"
            }
        else:
            source_names = [normalized(str(source.get("drug_name", ""))) for source in sources]
            passed = (
                passed and bool(sources)
                and all(str(source.get("source", "")).lower().endswith(".pdf") for source in sources)
                and all(source.get("page") not in (None, "") for source in sources)
                and all(source.get("score") is not None for source in sources)
                and all(normalized(brand) in name for name in source_names)
                and all(all(normalized(token) in name for token in detail_tokens) for name in source_names)
            )
        results.append({
            "query": query, "group": "medicine", "passed": passed, "elapsed_seconds": round(elapsed, 3),
            "answer_mode": payload.get("answer_mode"), "source_count": len(sources),
            "fallback_reason": payload.get("retrieval_stats", {}).get("fallback_reason"),
        })

    for query, expected_mode in SAFETY_CASES:
        started = time.perf_counter()
        response = session.post(f"{args.base_url}/api/ask", json={"question": query}, timeout=40)
        elapsed = time.perf_counter() - started
        response.raise_for_status()
        payload = response.json()
        stats = payload.get("retrieval_stats", {})
        passed = bool(payload.get("answer")) and payload.get("answer_mode") == expected_mode
        if expected_mode in {"symptom_triage", "emergency_triage"}:
            passed = passed and not stats.get("faiss_called") and not stats.get("llm_called") and not payload.get("sources")
        if expected_mode == "emergency_triage":
            passed = passed and ("112" in payload["answer"] or "acil" in normalized(payload["answer"]))
        if "bulunmayan" in query:
            passed = passed and not payload.get("sources") and stats.get("fallback_reason") in {
                "document_not_found", "medicine_not_found"
            }
        results.append({
            "query": query, "group": "safety", "passed": passed, "elapsed_seconds": round(elapsed, 3),
            "answer_mode": payload.get("answer_mode"), "source_count": len(payload.get("sources", [])),
            "fallback_reason": stats.get("fallback_reason"),
        })

    durations = [item["elapsed_seconds"] for item in results]
    summary = {
        "total": len(results),
        "passed": sum(item["passed"] for item in results),
        "failed": sum(not item["passed"] for item in results),
        "average_seconds": round(statistics.mean(durations), 3),
        "first_query_seconds": durations[0],
        "median_seconds": round(statistics.median(durations), 3),
        "max_seconds": max(durations),
        "fallback_counts": dict(__import__("collections").Counter(item["fallback_reason"] or "none" for item in results)),
        "failures": [item for item in results if not item["passed"]],
        "results": results,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
