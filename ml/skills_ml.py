import os
import json
import joblib
from typing import List

# ---------------- MODEL PATHS ----------------

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "skills_model.pkl")
MLB_PATH = os.path.join(MODELS_DIR, "skills_mlb.pkl")

KEYWORDS_FILE = os.path.join(os.path.dirname(__file__), "skills_keywords.json")

_model = None
_mlb = None


# ---------------- LOAD MODEL ----------------

def _load_model():
    """Load ML model & label binarizer."""
    global _model, _mlb
    if _model is None or _mlb is None:
        _model = joblib.load(MODEL_PATH)
        _mlb = joblib.load(MLB_PATH)
    return _model, _mlb


# ---------------- LOAD KEYWORD RULES ----------------

with open(KEYWORDS_FILE, "r", encoding="utf-8") as f:
    RULE_KEYWORDS = json.load(f)


# ---------------- RULE-BASED EXTRACTION ----------------

def _rule_based_skills(text: str) -> List[str]:
    """High-precision keyword-based detection."""
    t = text.lower()
    found = []

    for skill, words in RULE_KEYWORDS.items():
        for w in words:
            if w in t:
                found.append(skill)
                break

    return sorted(set(found))


# ---------------- ML FALLBACK ----------------

def _ml_skills(text: str) -> List[str]:
    """ML prediction (only used when rule-based fails)."""
    if not text.strip():
        return []

    model, mlb = _load_model()
    proba = model.predict_proba([text])[0]

    THRESHOLD = 0.32
    indices = [i for i, p in enumerate(proba) if p >= THRESHOLD]

    if not indices:  # choose top 2 for safety
        indices = sorted(range(len(proba)), key=lambda i: proba[i], reverse=True)[:2]

    return [mlb.classes_[i] for i in indices]


# ---------------- PUBLIC API FUNCTIONS ----------------

def predict_skills_for_text(text: str) -> List[str]:
    """Google-level hybrid classifier."""
    text = text.lower().strip()

    rule = _rule_based_skills(text)
    if rule:
        return rule  # RULES ALWAYS WIN

    return _ml_skills(text)


def predict_skills_for_many(texts: List[str]) -> dict:
    """Aggregate skills for multiple reflections."""
    from collections import Counter

    counter = Counter()
    for t in texts:
        skills = predict_skills_for_text(t)
        for s in skills:
            counter[s] += 1

    return dict(counter)
