# =============================
# reflection_mongo.py (FINAL MERGED)
# =============================

import os
import faiss
import numpy as np
import calendar
import requests
from difflib import SequenceMatcher
from bson import ObjectId
from datetime import datetime, timedelta, date
from fastapi import APIRouter, HTTPException, Form, Query

# Embedding model
from sentence_transformers import SentenceTransformer

# Groq (AI Search Assistant)
from groq import Groq

# App Imports
from app.db import get_conn, get_reflection_collection
from app.utils import authorize
from app.config import OLLAMA_URL, AI_MODEL

# ML Imports
from ml.skills_ml import predict_skills_for_text
from ml.sentiment import analyze_sentiment
from app.ml_routes import generate_skill_analysis

# -----------------------------
# CONSTANTS
# -----------------------------

FAISS_INDEX_PATH = "faiss.index"
VECTOR_DIM = 768
CHUNK_SIZE = 350
CHUNK_OVERLAP = 60
EMBED_MODEL_LOCAL = "all-mpnet-base-v2"

router = APIRouter(prefix="/api/reflection", tags=["Reflections (MongoDB)"])

# Embedding model
sentence_model = SentenceTransformer(EMBED_MODEL_LOCAL)

# Groq Client setup
try:
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    GROQ_MODEL = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
except:
    groq_client = None
    GROQ_MODEL = "llama3-70b-8192"

# -----------------------------
# FAISS INITIALIZATION
# -----------------------------

def load_faiss():
    if os.path.exists(FAISS_INDEX_PATH):
        return faiss.read_index(FAISS_INDEX_PATH)
    return faiss.IndexFlatL2(VECTOR_DIM)

faiss_index = load_faiss()
mongo_db = get_reflection_collection().database
payload_collection = mongo_db["faiss_payloads"]

def save_payload(faiss_id: int, payload: dict):
    payload["_id"] = faiss_id
    payload_collection.replace_one({"_id": faiss_id}, payload, upsert=True)

def load_payload(faiss_id: int):
    return payload_collection.find_one({"_id": faiss_id})

# -----------------------------
# HELPERS
# -----------------------------

def chunk_text(text: str):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + CHUNK_SIZE])
        if len(chunk.strip()) > 20:
            chunks.append(chunk)
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def embed_text(text: str):
    return sentence_model.encode(text or "").astype("float32").tolist()

def generate_reflection_id():
    col = get_reflection_collection()
    last = col.find_one({"ReflectionID": {"$exists": True}}, sort=[("ReflectionID", -1)])
    if not last:
        return "RF-000001"
    return f"RF-{int(last['ReflectionID'].split('-')[1]) + 1:06d}"

def text_similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


# -----------------------------
# FIXED QUESTIONS
# -----------------------------
QUESTIONS = {
    "daily": [
        "What went well today and what did you learn?",
        "What didn't go well today and what did you learn?",
        "What are you planning to do or achieve tomorrow?"
    ],
    "weekly": [
        "What went well this week and what did you learn?",
        "What didn't go well this week and what did you learn?",
        "What are you planning to do or achieve next week?"
    ],
    "monthly": [
        "What went well this month and what did you learn?",
        "What didn't go well this month and what did you learn?",
        "What are you planning to do or achieve next month?"
    ]
}

# -----------------------------
# GET QUESTIONS
# -----------------------------
@router.get("/questions")
def get_questions(period: str = Query("daily")):
    if period not in QUESTIONS:
        raise HTTPException(400, "Invalid reflection period")
    return {"status": "success", "period": period, "questions": QUESTIONS[period]}

# -----------------------------
# SUBMIT REFLECTION
# -----------------------------
@router.post("/submit")
def submit_reflection(
    email: str = Form(...),
    period: str = Form(...),
    selected_date: str = Form(...),
    q1: str = Form(...),
    q2: str = Form(...),
    q3: str = Form(...)
):
    global faiss_index

    # Normalize period
    period = period.lower().strip()
    if period not in QUESTIONS:
        raise HTTPException(400, f"Invalid period '{period}'")

    # -----------------------------
    # DATE PARSING (original logic)
    # -----------------------------
    week_start = week_end = None
    month_name = None
    year_val = None

    try:
        if period == "daily":
            selected_dt_date = datetime.fromisoformat(selected_date).date()

        elif period == "weekly":
            y, w = selected_date.split("-W")
            year_val, week_int = int(y), int(w)
            selected_dt_date = date.fromisocalendar(year_val, week_int, 1)
            week_start = datetime.combine(selected_dt_date, datetime.min.time())
            week_end = datetime.combine(selected_dt_date + timedelta(days=6), datetime.min.time())

        elif period == "monthly":
            year_val, month_val = map(int, selected_date.split("-"))
            selected_dt_date = date(year_val, month_val, 1)
            month_name = calendar.month_name[month_val]

    except:
        raise HTTPException(400, "Invalid selected_date format")

    if selected_dt_date > datetime.now().date():
        raise HTTPException(400, "Cannot submit reflection for future dates.")

    # -----------------------------
    # EMPLOYEE FETCH
    # -----------------------------
    conn = get_conn()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT ID, empID, Name, RoleID, ReportingManagerID FROM employees WHERE EmailID=%s", (email,))
    user = cur.fetchone()
    cur.close(); conn.close()

    if not user:
        raise HTTPException(404, "Employee not found")

    col = get_reflection_collection()

    if period == "weekly":
        selected_dt_datetime = week_start
    else:
        selected_dt_datetime = datetime.combine(selected_dt_date, datetime.min.time())

    # -----------------------------
    # DUPLICATE CHECK
    # -----------------------------
    if col.find_one({"Email": email, "PeriodType": period, "SelectedDate": selected_dt_datetime}):
        return {"status": "failed", "message": f"You already submitted a {period} reflection for this date."}

    # -----------------------------
    # ML: SKILLS + SENTIMENT
    # -----------------------------
    q1_answers = q1.split(" | ")
    q2_answers = q2.split(" | ")
    q3_answers = q3.split(" | ")

    q1_text = " ".join(q1_answers)
    q2_text = " ".join(q2_answers)
    q3_text = " ".join(q3_answers)

    q1_skills = predict_skills_for_text(q1_text)
    q1_skill_scores = {s: 1.0 for s in q1_skills}

    q1_sentiment = analyze_sentiment(q1_text)
    q2_sentiment = analyze_sentiment(q2_text)
    q3_sentiment = analyze_sentiment(q3_text)

    # -----------------------------
    # INSERT REFLECTION
    # -----------------------------
    new_id = generate_reflection_id()

    payload = {
        "ReflectionID": new_id,
        "EmployeeID": user["ID"],
        "empID": user["empID"],
        "Email": email,
        "Name": user["Name"],
        "RoleID": user["RoleID"],
        "ReportingManagerID": user["ReportingManagerID"],
        "PeriodType": period,
        "SelectedDate": selected_dt_datetime,

        "Answers": {
            "Question1": {
                "question": QUESTIONS[period][0],
                "answers": q1_answers,
                "skills": q1_skills,
                "skills_score": q1_skill_scores,
                "sentiment": q1_sentiment
            },
            "Question2": {
                "question": QUESTIONS[period][1],
                "answers": q2_answers,
                "sentiment": q2_sentiment
            },
            "Question3": {
                "question": QUESTIONS[period][2],
                "answers": q3_answers,
                "sentiment": q3_sentiment
            },
        },

        "DateSubmitted": datetime.now().strftime("%Y-%m-%d"),
        "CreatedAt": datetime.utcnow()
    }

    if period == "weekly":
        payload["WeekStart"] = week_start
        payload["WeekEnd"] = week_end

    if period == "monthly":
        payload["MonthName"] = month_name
        payload["Year"] = year_val

    # --------------------------------------
    # PERIOD-WISE SKILL ANALYSIS (mySkills)
    # --------------------------------------
    reflection_query = {"Email": email, "PeriodType": period}

    if period == "daily":
        reflection_query["SelectedDate"] = selected_dt_datetime

    elif period == "weekly":
        reflection_query["SelectedDate"] = {"$gte": week_start, "$lte": week_end}

    elif period == "monthly":
        first = datetime(year_val, selected_dt_date.month, 1)
        last = datetime(year_val, selected_dt_date.month,
                        calendar.monthrange(year_val, selected_dt_date.month)[1])
        reflection_query["SelectedDate"] = {"$gte": first, "$lte": last}

    docs = list(col.find(reflection_query))

    full_texts = []
    for d in docs:
        ans = d.get("Answers", {})
        for q in ["Question1", "Question2", "Question3"]:
            full_texts.extend(ans.get(q, {}).get("answers", []))

    full_texts.extend(q1_answers + q2_answers + q3_answers)

    payload["mySkills"] = generate_skill_analysis(full_texts)

    col.insert_one(payload)

    # -----------------------------
    # FAISS VECTOR INDEXING
    # -----------------------------
    reflection_obj = {
        "ReflectionID": new_id,
        "EmployeeID": user["ID"],
        "Name": user["Name"],
        "Email": email,
        "PeriodType": period,
        "DateSubmitted": payload["DateSubmitted"],
        "Answers": {
            "Question1": {"question": QUESTIONS[period][0], "answers": q1_answers},
            "Question2": {"question": QUESTIONS[period][1], "answers": q2_answers},
            "Question3": {"question": QUESTIONS[period][2], "answers": q3_answers}
        }
    }

    new_vectors = []
    new_payloads = []

    for q_key in ["Question1", "Question2", "Question3"]:
        q_data = reflection_obj["Answers"][q_key]
        for ans in q_data["answers"]:
            full_text = f"Question: {q_data['question']}\nAnswer: {ans}"
            for chunk in chunk_text(full_text):
                vec = embed_text(chunk)
                new_vectors.append(vec)

                new_payloads.append({
                    "reflection_id": new_id,
                    "employee_name": user["Name"],
                    "employee_email": email,
                    "date_submitted": payload["DateSubmitted"],
                    "period": period,
                    "question": q_data["question"],
                    "full_answer": ans,
                    "chunk_text": chunk
                })

    if new_vectors:
        new_vectors = np.array(new_vectors, dtype="float32")
        start_idx = faiss_index.ntotal
        faiss_index.add(new_vectors)

        for i, p in enumerate(new_payloads):
            save_payload(start_idx + i, p)

        faiss.write_index(faiss_index, FAISS_INDEX_PATH)

    return {"status": "success", "ReflectionID": new_id, "message": "Reflection submitted successfully ✔️"}

# -----------------------------
# AI SEARCH /ask
# -----------------------------
@router.post("/ask")
def ask_reflection(question: str = Form(...), top_k: int = Form(6)):
    global faiss_index

    query_vec = np.array([embed_text(question)], dtype="float32")

    if faiss_index.ntotal == 0:
        return {"answer": "No reflections indexed.", "sources_found": 0, "relevant_reflections": []}

    distances, indices = faiss_index.search(query_vec, top_k)

    relevant_chunks = []
    for score, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        payload = load_payload(int(idx))
        if payload:
            relevant_chunks.append({
                "name": payload["employee_name"],
                "email": payload["employee_email"],
                "date": payload["date_submitted"],
                "question": payload["question"],
                "answer": payload["full_answer"],
                "score": float(score)
            })

    if not relevant_chunks:
        return {"answer": "No similar reflections found.", "sources_found": 0, "relevant_reflections": []}

    relevant_chunks.sort(key=lambda x: x["score"])
    context = "\n".join([
        f"{i+1}. {c['name']} ({c['date']})\nQ: {c['question']}\nA: {c['answer']}\n"
        for i, c in enumerate(relevant_chunks)
    ])

    system_prompt = f"""
You are a reflective assistant helping employees learn from past reflections.

User query: "{question}"

Context:
{context}

Respond professionally and only based on provided reflections.
"""

    if groq_client:
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.3,
                max_tokens=800
            )
            answer = response.choices[0].message.content.strip()
        except:
            answer = "AI service unavailable."
    else:
        answer = "AI service not configured."

    return {
        "answer": answer,
        "sources_found": len(relevant_chunks),
        "relevant_reflections": relevant_chunks
    }

# -----------------------------
# UPDATE REFLECTION
# -----------------------------
@router.patch("/update")
def update_reflection(
    reflection_id: str = Form(...),
    q1: str = Form(None),
    q2: str = Form(None),
    q3: str = Form(None)
):
    col = get_reflection_collection()
    if not col.find_one({"ReflectionID": reflection_id}):
        raise HTTPException(404, "Reflection not found")

    update_block = {}
    if q1: update_block["Answers.Question1.answers"] = q1.split(" | ")
    if q2: update_block["Answers.Question2.answers"] = q2.split(" | ")
    if q3: update_block["Answers.Question3.answers"] = q3.split(" | ")

    if not update_block:
        return {"status": "ignored", "message": "No changes provided"}

    col.update_one({"ReflectionID": reflection_id}, {"$set": update_block})
    return {"status": "success", "message": "Reflection updated"}

# -----------------------------
# DELETE REFLECTION
# -----------------------------
@router.delete("/delete")
def delete_reflection(reflection_id: str):
    col = get_reflection_collection()
    res = col.delete_one({"ReflectionID": reflection_id})
    if res.deleted_count == 0:
        raise HTTPException(404, "Reflection not found")
    return {"status": "success", "message": "Reflection deleted"}

# -----------------------------
# VIEW SELF REFLECTIONS
# -----------------------------
@router.get("/view")
def view_reflections(email: str, period: str = "daily"):
    conn = get_conn()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT ID FROM employees WHERE EmailID=%s", (email,))
    emp = cur.fetchone()
    cur.close(); conn.close()

    if not emp:
        raise HTTPException(404, "User not found")

    col = get_reflection_collection()
    records = list(col.find({"EmployeeID": emp["ID"], "PeriodType": period}, {"_id": 0}))
    return {"status": "success", "records": records}

# -----------------------------
# HIERARCHY VIEW
# -----------------------------
@router.get("/hierarchy")
def hierarchy_view(email: str, period: str = "daily"):

    conn = get_conn()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT ID, RoleID FROM employees WHERE EmailID=%s", (email,))
    u = cur.fetchone()
    cur.close(); conn.close()

    if not u:
        raise HTTPException(404, "User not found")

    col = get_reflection_collection()
    uid = u["ID"]; role = u["RoleID"]

    if role == 1:
        conn = get_conn(); cur = conn.cursor(dictionary=True)
        cur.execute("SELECT ID FROM employees WHERE ReportingManagerID=%s AND RoleID=2", (uid,))
        team = [r["ID"] for r in cur.fetchall()]
        cur.close(); conn.close()
        query = {"EmployeeID": {"$in": [uid] + team}, "PeriodType": period}

    elif role == 2:
        query = {"$or": [{"EmployeeID": uid}, {"ReportingManagerID": uid}], "PeriodType": period}

    else:
        query = {"EmployeeID": uid, "PeriodType": period}

    records = list(col.find(query, {"_id": 0}))
    return {"status": "success", "records": records}

@router.get("/team_skills")
def team_skills_report(
    email: str = Query(..., description="Manager/SBU/employee email"),
    period: str = Query(..., regex="^(daily|weekly|monthly)$"),
    date_value: str = Query(..., description="daily=YYYY-MM-DD, weekly=YYYY-W##, monthly=YYYY-MM"),
    target_email: str = Query(None, description="Skill report for this employee")
):
    """
    Returns SKILL ANALYSIS for:
    - Employee     → Only their skills
    - Manager      → Their team skills
    - SBU Head     → All teams under SBU
 
    If target_email is given → ONLY that employee's skills returned.
    """
 
    # STEP 1 — Identify logged-in user
    conn = get_conn()
    cur = conn.cursor(dictionary=True)
 
    cur.execute(
        "SELECT ID, Name, EmailID, RoleID FROM employees WHERE EmailID=%s",
        (email,)
    )
    u = cur.fetchone()
    cur.close()
    conn.close()
 
    if not u:
        raise HTTPException(404, "User not found")
 
    user_id = u["ID"]
    user_role = u["RoleID"]
 
    # STEP 2 — Build hierarchy (same as reflections/timesheet)
    team_ids = [user_id]  # include self
 
    conn = get_conn()
    cur = conn.cursor(dictionary=True)
 
    if user_role == 1:  # SBU HEAD
        # Level 2 → Managers
        cur.execute("""
            SELECT ID FROM employees
            WHERE ReportingManagerID=%s AND RoleID=2
        """, (user_id,))
        managers = [m["ID"] for m in cur.fetchall()]
        team_ids.extend(managers)
 
        # Level 3 → Employees under managers
        if managers:
            q = "SELECT ID FROM employees WHERE ReportingManagerID IN (%s)" % ",".join(["%s"] * len(managers))
            cur.execute(q, managers)
            employees = [e["ID"] for e in cur.fetchall()]
            team_ids.extend(employees)
 
    elif user_role == 2:  # MANAGER
        cur.execute(
            "SELECT ID FROM employees WHERE ReportingManagerID=%s",
            (user_id,)
        )
        employees = [e["ID"] for e in cur.fetchall()]
        team_ids.extend(employees)
 
    # (Employee role → only self remains)
 
    cur.close()
    conn.close()
 
    # STEP 3 — Build date filter (same as reflections)
    col = get_reflection_collection()
 
    try:
        if period == "daily":
            sel_dt = datetime.fromisoformat(date_value).replace(hour=0, minute=0, second=0)
            date_filter = {"SelectedDate": sel_dt}
 
        elif period == "weekly":
            y, w = date_value.split("-W")
            week_start = date.fromisocalendar(int(y), int(w), 1)
            week_end = week_start + timedelta(days=6)
            date_filter = {
                "SelectedDate": {
                    "$gte": datetime.combine(week_start, datetime.min.time()),
                    "$lte": datetime.combine(week_end, datetime.min.time())
                }
            }
 
        else:  # monthly
            year, month = map(int, date_value.split("-"))
            first = datetime(year, month, 1)
            last = datetime(year, month, calendar.monthrange(year, month)[1])
            date_filter = {
                "SelectedDate": {"$gte": first, "$lte": last}
            }
 
    except:
        raise HTTPException(400, "Invalid date_value format")
 
    # STEP 4 — Fetch reflections of all team members
    team_records = list(
        col.find(
            {
                "EmployeeID": {"$in": team_ids},
                "PeriodType": period,
                **date_filter
            },
            {"_id": 0}
        )
    )
 
    if not team_records:
        return {
            "status": "empty",
            "message": "No reflections found for this period",
            "skills": None
        }
 
    # STEP 5 — Aggregate skills PER employee
    skill_output = {}
 
    for member_id in set([r["EmployeeID"] for r in team_records]):
        member_docs = [doc for doc in team_records if doc["EmployeeID"] == member_id]
 
        # Merge all reflection text
        all_texts = []
        for d in member_docs:
            answers = d.get("Answers", {})
            for q in ["Question1", "Question2", "Question3"]:
                all_texts.extend(answers.get(q, {}).get("answers", []))
 
        # Run skill analysis (your existing AI function)
        analysis = generate_skill_analysis(all_texts)
 
        # Get employee name/email
        conn = get_conn()
        cur = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT Name, EmailID FROM employees WHERE ID=%s",
            (member_id,)
        )
        member = cur.fetchone()
        cur.close()
        conn.close()
 
        skill_output[member_id] = {
            "name": member["Name"],
            "email": member["EmailID"],
            "total_reflections": len(member_docs),
            "skills": analysis
        }
 
    # STEP 6 — If specific employee requested (target_email)
    if target_email:
        for m in skill_output.values():
            if m["email"] == target_email:
                return {
                    "status": "success",
                    "period": period,
                    "date_value": date_value,
                    "skills": m  # return only that employee
                }
 
        return {
            "status": "empty",
            "message": "No skills found for this employee",
            "skills": None
        }
 
    # STEP 7 — Default return (full team)
    return {
        "status": "success",
        "period": period,
        "date_value": date_value,
        "team_members": skill_output
    }
# -----------------------------
# DISABLED MANUAL ADD
# -----------------------------
@router.post("/add_answer")
def disabled_add_answer():
    return {"status": "blocked", "message": "Manual answer updates disabled. Use /submit instead."}


@router.get("/reminder")
def reflection_reminder(
    email: str = Query(...),
    period: str = Query(..., regex="^(daily|weekly|monthly)$"),
    selected_date: str = Query(...)
):
    """
    Returns the previous period's Question3 answers (daily/weekly/monthly)
    in a professional, personalized format.
    """
    col = get_reflection_collection()
 
    # ---------------------------------
    # 1. Fetch user name from MySQL
    # ---------------------------------
    conn = get_conn()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT Name FROM employees WHERE EmailID=%s", (email,))
    user = cur.fetchone()
    cur.close(); conn.close()
 
    if not user:
        raise HTTPException(404, "User not found")
 
    user_name = user["Name"]
 
    # -------------------------------
    # DAILY — YYYY-MM-DD
    # -------------------------------
    if period == "daily":
        try:
            today = datetime.fromisoformat(selected_date).date()
        except:
            raise HTTPException(400, "Daily format must be YYYY-MM-DD")
 
        prev_date = today - timedelta(days=1)
        prev_dt = datetime.combine(prev_date, datetime.min.time())
 
        query = {
            "Email": email,
            "PeriodType": "daily",
            "SelectedDate": prev_dt
        }
 
        period_text = "yesterday"
 
    # -------------------------------
    # WEEKLY — YYYY-W##
    # UPDATED BLOCK ✔
    # -------------------------------
    elif period == "weekly":
        if "-W" not in selected_date:
            raise HTTPException(400, "Weekly format must be YYYY-W##")
 
        try:
            year, week = selected_date.split("-W")
            year = int(year)
            week = int(week)
        except:
            raise HTTPException(400, "Invalid weekly format")
 
        # Current week Monday
        curr_week_start = date.fromisocalendar(year, week, 1)
 
        # Previous week Monday
        prev_week_start = curr_week_start - timedelta(days=7)
 
        # ✔ FIX: Weekly is stored as EXACT previous Monday (NOT range)
        prev_dt = datetime.combine(prev_week_start, datetime.min.time())
 
        query = {
            "Email": email,
            "PeriodType": "weekly",
            "SelectedDate": prev_dt
        }
 
        period_text = "last week"
 
    # -------------------------------
    # MONTHLY — YYYY-MM
    # -------------------------------
    else:  # monthly
        try:
            year, month = map(int, selected_date.split("-"))
        except:
            raise HTTPException(400, "Monthly format must be YYYY-MM")
 
        if month == 1:
            prev_year = year - 1
            prev_month = 12
        else:
            prev_year = year
            prev_month = month - 1
 
        prev_start = datetime(prev_year, prev_month, 1)
        last_day = calendar.monthrange(prev_year, prev_month)[1]
        prev_end = datetime(prev_year, prev_month, last_day)
 
        query = {
            "Email": email,
            "PeriodType": "monthly",
            "SelectedDate": {
                "$gte": prev_start,
                "$lte": prev_end
            }
        }
 
        period_text = "last month"
 
    # -------------------------------
    # FETCH PREVIOUS Q3 ANSWERS
    # -------------------------------
    prev_ref = col.find_one(query)
 
    if not prev_ref:
        return {
            "status": "empty",
            "hasReminder": False,
            "message": "No previous reflection found"
        }
 
    q3 = prev_ref.get("Answers", {}).get("Question3", {}).get("answers", [])
 
    # -------------------------------
    # BUILD PERSONALIZED MESSAGE
    # -------------------------------
    msg = f"Hey {user_name}, here is what you planned {period_text}:"
 
    return {
        "status": "success",
        "hasReminder": True,
        "userMessage": msg,
        "previousPlans": q3
    }