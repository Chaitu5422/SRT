# app/ai.py
from fastapi import APIRouter, Request
import requests
from app.config import OLLAMA_URL, AI_MODEL


router = APIRouter(prefix="/api/ai", tags=["AI Assistant"])


# ----------------- BASE CALL FUNCTION -----------------
def ask_ai(prompt: str):
    payload = {
        "model": AI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    data = response.json()

    if "message" in data:
        return data["message"]["content"]
    return "❌ No response received from LLAMA model. Check if Ollama is running."


# -----------------------------------------------------------
# ⭐ REPHRASE TEXT — CLEAN OUTPUT, ONLY REWRITTEN SENTENCE
# -----------------------------------------------------------
@router.get("/rephrase")
@router.post("/rephrase")
async def rephrase_text(request: Request, text: str = None):

    # Case 1 → GET /rephrase?text=xxx
    if text:
        clean_text = text

    # Case 2 → POST JSON: { "text": "..." }
    else:
        try:
            body = await request.json()
            clean_text = body.get("text")
        except:
            clean_text = None

    if not clean_text:
        return {"error": "No text provided"}

    # 🔥 Strong instruction to prevent explanations
    prompt = f"""
Rewrite the text below to sound more professional and concise.
Return ONLY the rewritten text — no explanation, no intro, no formatting.

Text:
{clean_text}
"""

    rephrased = ask_ai(prompt).strip()

    # 🔥 Safety filter (removes unwanted intro if model tries again)
    blocked_phrases = [
        "Here is a rewritten version",
        "Rewrite the following reflection",
        "More professional version",
        "Rewritten text",
        ":",
    ]
    for phrase in blocked_phrases:
        if rephrased.lower().startswith(phrase.lower()):
            rephrased = rephrased[len(phrase):].strip("-: ").strip()

    return {
        "original": clean_text,
        "rephrased": rephrased
    }


# -----------------------------------------------------------
# SUGGEST REFLECTION POINTS
# -----------------------------------------------------------
@router.post("/suggest")
def suggest_points(question_type: str, text: str = ""):
    prompt = f"""
    Suggest meaningful reflection points for **{question_type}**.
    Based on user input if provided:

    {text or "No previous context provided"}
    """
    return {"suggestions": ask_ai(prompt)}


# -----------------------------------------------------------
# WEEKLY SUMMARY
# -----------------------------------------------------------
@router.post("/weekly-summary")
def weekly_summary(entries: list):
    merged = "\n".join(f"- {e}" for e in entries)
    prompt = f"Create a structured weekly reflection summary based on:\n{merged}"
    return {"weekly_summary": ask_ai(prompt)}


# -----------------------------------------------------------
# MONTHLY SUMMARY
# -----------------------------------------------------------
@router.post("/monthly-summary")
def monthly_summary(weeks: list, month: str):
    combined = "\n\n".join([f"Week {i+1}:\n{w}" for i, w in enumerate(weeks)])
    prompt = f"Generate a professional monthly reflection summary for **{month}**.\n\n{combined}"
    return {"monthly_summary": ask_ai(prompt)}