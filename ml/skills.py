# backend/ml/skills.py
import re
from typing import List, Dict

# You can expand/modify this as you like
SKILL_KEYWORDS: Dict[str, list[str]] = {
    "python": ["python", "py "],
    "sql": ["sql", "query", "database"],
    "api development": ["api", "endpoint", "rest api", "fastapi"],
    "testing": ["test case", "unit test", "testing", "debug"],
    "communication": ["meeting", "presentation", "call", "communicat"],
}


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return text


def extract_skills(text: str) -> List[str]:
    """
    Very simple keyword based skill extraction.
    Input  : one reflection text
    Output : list like ["python", "sql"]
    """
    if not text:
        return []

    text = _normalize(text)
    found = set()

    for skill_name, patterns in SKILL_KEYWORDS.items():
        for p in patterns:
            if p in text:
                found.add(skill_name)
                break

    return sorted(found)


def extract_skills_from_many(reflections: List[str]) -> Dict[str, int]:
    """
    Take multiple reflection texts and count how many times each skill appears.
    Useful for 'one month reflection → skill summary'
    """
    counter: Dict[str, int] = {}
    for txt in reflections:
        skills = extract_skills(txt)
        for s in skills:
            counter[s] = counter.get(s, 0) + 1
    return counter
