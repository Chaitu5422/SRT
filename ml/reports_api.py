# app/reports_api.py
from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime, timedelta
from gspread import authorize as gspread_auth     # Google Sheets authorize (renamed)
from pydantic import BaseModel
from typing import Optional
from ml.ml_pipeline import run_pipeline_for_period
from app.db import get_ml_reports_collection      # NEW: from db.py
from app.utils import authorize                   # JWT AUTH

router = APIRouter(prefix="/api/reports", tags=["ML Reports"])


# ------------------------------
# Request Model for Trigger API
# ------------------------------
class TriggerRequest(BaseModel):
    email: str
    period: str = "weekly"
    start_date: Optional[str] = None
    end_date: Optional[str] = None


# ------------------------------
# Trigger ML Pipeline
# ------------------------------
@router.post("/trigger")
def trigger_pipeline(payload: TriggerRequest):
    now = datetime.utcnow()

    # Determine the period range
    if payload.start_date and payload.end_date:
        start = datetime.fromisoformat(payload.start_date)
        end = datetime.fromisoformat(payload.end_date)
    else:
        if payload.period == "weekly":
            end = now
            start = now - timedelta(days=7)
        elif payload.period == "monthly":
            end = now
            start = now - timedelta(days=30)
        else:
            raise HTTPException(status_code=400, detail="Unsupported period")

    result = run_pipeline_for_period(start, end, payload.period, email=payload.email)

    return {"status": "ok", "report_id": result["report_id"]}


# ------------------------------
# LIST ALL ML REPORTS
# Protected with JWT
# ------------------------------
@router.get("/")
def list_reports(limit: int = 20):
    coll = get_ml_reports_collection()

    docs = list(coll.find().sort("generated_at", -1).limit(limit))
    out = []

    for d in docs:
        out.append({
            "id": str(d.get("_id")),
            "email": d.get("email"),
            "period_label": d.get("period_label"),
            "period_start": d.get("period_start"),
            "period_end": d.get("period_end"),
            "generated_at": d.get("generated_at"),
        })

    return out


# ------------------------------
# GET SINGLE REPORT
# ------------------------------
@router.get("/{report_id}")
def get_report(report_id: str):
    from bson import ObjectId

    coll = get_ml_reports_collection()

    doc = coll.find_one({"_id": ObjectId(report_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Report not found")

    return {
        "id": str(doc.get("_id")),
        "email": doc.get("email"),
        "period_label": doc.get("period_label"),
        "period_start": doc.get("period_start"),
        "period_end": doc.get("period_end"),
        "generated_at": doc.get("generated_at"),
        "summary": doc.get("summary")
    }
