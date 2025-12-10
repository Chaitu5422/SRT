from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import List, Dict
import os, json
import calendar
from datetime import datetime, timedelta, date

# MongoDB
from app.db import get_reflection_collection

# ML Imports
from ml.sentiment import analyze_sentiment
from ml.skills_ml import predict_skills_for_text, predict_skills_for_many

router = APIRouter(prefix="/api/ml", tags=["Machine Learning"])


# -----------------------------
# MODELS
# -----------------------------
class TextBody(BaseModel):
    text: str

class ReflectionListBody(BaseModel):
    reflections: List[str]

class AnalysisBody(BaseModel):
    reflections: List[str]


# -----------------------------
# Load skill categories file
# -----------------------------
SKILLS_JSON_PATH = os.path.join(os.path.dirname(__file__), "../ml/skills_keywords.json")
SKILLS_JSON_PATH = os.path.abspath(SKILLS_JSON_PATH)

with open(SKILLS_JSON_PATH, "r", encoding="utf-8") as f:
    SKILL_CATEGORIES: Dict[str, List[str]] = json.load(f)


# ============================================================
# 🔥 REUSABLE SKILL ANALYSIS (Used in: Submit + API)
# ============================================================
def generate_skill_analysis(reflection_texts: List[str]):
    if not reflection_texts:
        return {
            "all_skills": {},
            "skill_levels": {},
            "categories": {},
            "trend": {},
            "top_skills": [],
            "skills_to_improve": [],
            "insights": []
        }

    # ---------------------------------------------------------
    # Extract skills per reflection
    # ---------------------------------------------------------
    reflection_skills = [predict_skills_for_text(text) for text in reflection_texts]

    # Flatten list
    all_skills = []
    for s in reflection_skills:
        all_skills.extend(s)

    # Frequency calculation
    freq = {}
    for s in all_skills:
        freq[s] = freq.get(s, 0) + 1

    # ---------------------------------------------------------
    # Category grouping
    # ---------------------------------------------------------
    categories = {}
    for category, keywords in SKILL_CATEGORIES.items():
        matched = [skill for skill in freq if skill in keywords or skill == category]
        if matched:
            categories[category] = matched

    # ---------------------------------------------------------
    # Skill levels (1–5 scale)
    # ---------------------------------------------------------
    skill_levels = {}
    max_freq = max(freq.values()) if freq else 1

    for skill, count in freq.items():
        level = round((count / max_freq) * 5)
        skill_levels[skill] = max(level, 1)

    # ---------------------------------------------------------
    # Trend detection
    # ---------------------------------------------------------
    timeline = {}
    for idx, sk_list in enumerate(reflection_skills):
        for s in sk_list:
            timeline.setdefault(s, []).append(idx)

    trend = {"improving": [], "declining": [], "stable": []}

    for skill, occ in timeline.items():
        if len(occ) <= 1:
            trend["stable"].append(skill)
        else:
            if occ[-1] > occ[0]:
                trend["improving"].append(skill)
            elif occ[-1] < occ[0]:
                trend["declining"].append(skill)
            else:
                trend["stable"].append(skill)

    # ---------------------------------------------------------
    # Top skills
    # ---------------------------------------------------------
    top_3 = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:3]
    top_skills = [x[0] for x in top_3]

    # ---------------------------------------------------------
    # Skills to improve
    # ---------------------------------------------------------
    skills_to_improve = [skill for skill, lvl in skill_levels.items() if lvl <= 2]

    # ---------------------------------------------------------
    # Insights
    # ---------------------------------------------------------
    insights = []

    if top_skills:
        insights.append(f"Top strengths: {', '.join(top_skills)}")

    if trend["improving"]:
        insights.append(f"Improving skills: {', '.join(trend['improving'])}")

    if trend["declining"]:
        insights.append(f"Declining skills: {', '.join(trend['declining'])}")

    if skills_to_improve:
        insights.append(f"Needs improvement: {', '.join(skills_to_improve)}")

    if not insights:
        insights.append("No major skill insights detected.")

    # ---------------------------------------------------------
    # Final output
    # ---------------------------------------------------------
    return {
        "all_skills": freq,
        "skill_levels": skill_levels,
        "categories": categories,
        "trend": trend,
        "top_skills": top_skills,
        "skills_to_improve": skills_to_improve,
        "insights": insights
    }


# ============================================================
# 1️⃣ Sentiment API
# ============================================================
@router.post("/sentiment")
def ml_sentiment(body: TextBody):
    return {"status": "success", "sentiment": analyze_sentiment(body.text)}


# ============================================================
# 2️⃣ Skill extract (single)
# ============================================================
@router.post("/skills_one")
def ml_skill_one(body: TextBody):
    return {"status": "success", "skills": predict_skills_for_text(body.text)}


# ============================================================
# 3️⃣ Skill extract (multiple)
# ============================================================
@router.post("/skills")
def ml_skill_many(body: ReflectionListBody):
    return {"status": "success", "skills": predict_skills_for_many(body.reflections)}


# ============================================================
# 4️⃣ Full Skill Analysis API
# ============================================================
@router.post("/skill_analysis")
def skill_analysis(body: AnalysisBody):
    output = generate_skill_analysis(body.reflections)
    return {
        "status": "success",
        "total_reflections": len(body.reflections),
        **output
    }


# ============================================================
# 5️⃣ DAILY / WEEKLY / MONTHLY TOP SKILLS API
# ============================================================
@router.get("/skill_analysis_period")
def skill_analysis_period(
    email: str = Query(...),
    period: str = Query(..., regex="^(daily|weekly|monthly)$"),
    date_value: str = Query(...)
):
    col = get_reflection_collection()

    # DAILY ---------------------------------------------------
    if period == "daily":
        selected_dt = datetime.fromisoformat(date_value).replace(hour=0, minute=0, second=0)
        query = {
            "Email": email,
            "PeriodType": "daily",
            "SelectedDate": selected_dt
        }

    # WEEKLY --------------------------------------------------
    elif period == "weekly":
        year, week = date_value.split("-W")
        year = int(year)
        week = int(week)

        week_start = date.fromisocalendar(year, week, 1)
        week_end = week_start + timedelta(days=6)

        query = {
            "Email": email,
            "PeriodType": "weekly",
            "SelectedDate": {
                "$gte": datetime.combine(week_start, datetime.min.time()),
                "$lte": datetime.combine(week_end, datetime.min.time())
            }
        }

    # MONTHLY -------------------------------------------------
    else:
        year, month = map(int, date_value.split("-"))
        first = datetime(year, month, 1)
        last_day = calendar.monthrange(year, month)[1]
        last = datetime(year, month, last_day)
        
        query = {
            "Email": email,
            "PeriodType": "monthly",
            "SelectedDate": {
                "$gte": first,
                "$lte": last
            }
        }

    # Fetch reflections for the given period
    docs = list(col.find(query))

    if not docs:
        return {"status": "empty", "message": "No reflections found for this period"}

    # Collect answers
    texts = []
    for d in docs:
        ans = d.get("Answers", {})
        for q in ["Question1", "Question2", "Question3"]:
            texts.extend(ans.get(q, {}).get("answers", []))

    # Analyze using main engine
    output = generate_skill_analysis(texts)

    return {
        "status": "success",
        "total_reflections": len(docs),
        "email": email,
        "period": period,
        "date": date_value,
        "top_3_skills": output["top_skills"],
        "analysis": output
    }
