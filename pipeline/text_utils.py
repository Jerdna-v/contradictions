import json
import os
import re
from typing import Dict, List

_CITE_RE = re.compile(r"\{\{cite:[a-f0-9]{40}\}\}")
_FIG_RE = re.compile(r"\b(fig\.|figure|table)\s*\d+\b", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = _CITE_RE.sub(" ", text)
    text = _FIG_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def extract_sections(body_text: List[Dict[str, str]]) -> Dict[str, str]:
    intro_parts = []
    results_parts = []
    conclusion_parts = []
    limitations_parts = []
    future_parts = []

    for item in body_text or []:
        heading = (item.get("heading") or "").lower()
        text = item.get("text") or ""
        if not text:
            continue

        if "introduction" in heading:
            intro_parts.append(text[:500])
        if any(key in heading for key in ["result", "experiment", "evaluation"]):
            results_parts.append(text)
        if any(key in heading for key in ["conclusion", "summary"]):
            conclusion_parts.append(text)
        if "limitation" in heading:
            limitations_parts.append(text)
        if any(key in heading for key in ["future", "discussion"]):
            future_parts.append(text)

    return {
        "intro_text": clean_text(" ".join(intro_parts)),
        "results_text": clean_text(" ".join(results_parts)),
        "conclusion_text": clean_text(" ".join(conclusion_parts)),
        "limitations_text": clean_text(" ".join(limitations_parts)),
        "future_work_text": clean_text(" ".join(future_parts)),
    }


def count_sentences(text: str) -> int:
    if not text:
        return 0
    parts = re.split(r"\.\s+|\.\n", text)
    return sum(1 for p in parts if p.strip())


def lexical_filter_enabled() -> bool:
    path = os.getenv("LEXICAL_DICT_PATH", "")
    return bool(path) and os.path.isfile(path)


def load_lexical_dict(path: str) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def matches_lexical_dict(text: str, entry: Dict[str, object]) -> bool:
    canonical = entry.get("canonical", "")
    aliases = entry.get("aliases", [])
    terms = [canonical] + list(aliases)
    for term in terms:
        if not term:
            continue
        if re.search(rf"\b{re.escape(term)}\b", text, re.IGNORECASE):
            return True
    return False
