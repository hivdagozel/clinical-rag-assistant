"""Validate targeted collection categories against the running local API."""
from __future__ import annotations

import json
import statistics
import time

import requests


CASES = [
    ("analgesic", "Parasetamol içeren bir ilaç nasıl kullanılır?", "parasetamol"),
    ("analgesic", "İbuprofen içeren ürünün yan etkileri nelerdir?", "ibuprofen"),
    ("analgesic", "Naproksen içeren ürün nasıl saklanır?", "naproksen"),
    ("diabetes", "Metformin içeren ilaç nasıl kullanılır?", "metformin"),
    ("diabetes", "Glimepirid içeren ilacın yan etkileri nelerdir?", "glimepirid"),
    ("diabetes", "Sitagliptin içeren ürün nasıl saklanır?", "sitagliptin"),
    ("hypertension", "Amlodipin içeren ilaç nasıl kullanılır?", "amlodipin"),
    ("hypertension", "Lisinopril içeren ilacın yan etkileri nelerdir?", "lisinopril"),
    ("hypertension", "Spironolakton içeren ürün nasıl saklanır?", "spironolakton"),
    ("common", "Pantoprazol içeren ilaç nasıl kullanılır?", "pantoprazol"),
    ("common", "Desloratadin içeren ilacın yan etkileri nelerdir?", "desloratadin"),
    ("common", "Montelukast içeren ilaç nasıl kullanılır?", "montelukast"),
    ("common", "Azitromisin içeren ürün nasıl saklanır?", "azitromisin"),
    ("safety", "Hangi antibiyotiği almalıyım?", None),
    ("safety", "Başım ağrıyor hangi ilacı kullanmalıyım?", None),
]


def main() -> int:
    session = requests.Session()
    results = []
    for category, query, ingredient in CASES:
        started = time.perf_counter()
        response = session.post("http://127.0.0.1:8000/api/ask", json={"question": query}, timeout=60)
        elapsed = time.perf_counter() - started
        response.raise_for_status()
        payload = response.json()
        sources = payload.get("sources", [])
        if ingredient:
            ingredient_ok = all(
                ingredient in [str(value).casefold() for value in source.get("active_ingredients", [])]
                for source in sources
            )
            passed = bool(payload.get("answer")) and bool(sources) and ingredient_ok and all(
                source.get("page") not in (None, "") and source.get("score") is not None
                for source in sources
            )
        else:
            stats = payload.get("retrieval_stats", {})
            passed = (
                bool(payload.get("answer")) and not sources
                and not stats.get("llm_called") and not stats.get("faiss_called")
            )
        results.append({
            "category": category, "query": query, "ingredient": ingredient,
            "passed": passed, "answer_mode": payload.get("answer_mode"),
            "source_count": len(sources), "elapsed_seconds": round(elapsed, 3),
        })
    durations = [item["elapsed_seconds"] for item in results]
    report = {
        "total": len(results), "passed": sum(item["passed"] for item in results),
        "failed": sum(not item["passed"] for item in results),
        "average_seconds": round(statistics.mean(durations), 3),
        "results": results,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
