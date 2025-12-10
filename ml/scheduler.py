# app/scheduler.py
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime, timedelta
import pytz
import logging

from ml.ml_pipeline import run_pipeline_for_period

logger = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")

def start_scheduler(app):
    scheduler = BackgroundScheduler(timezone=IST)
    # Run nightly: compute embeddings & produce daily/weekly summaries as needed
    # Example: generate weekly summary every Monday at 02:30 IST
    scheduler.add_job(
        func=lambda: _run_weekly(),
        trigger=CronTrigger(day_of_week="mon", hour=2, minute=30),
        id="weekly_summary",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=3600
    )
    # Monthly summary: run 1st day of month at 03:00 IST
    scheduler.add_job(
        func=lambda: _run_monthly(),
        trigger=CronTrigger(day="1", hour=3, minute=0),
        id="monthly_summary",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=3600
    )

    scheduler.start()

    @app.on_event("shutdown")
    def _shutdown():
        scheduler.shutdown()

def _run_weekly():
    now = datetime.utcnow()
    start = now - timedelta(days=7)
    try:
        res = run_pipeline_for_period(start, now, "weekly")
        logger.info("Weekly ML report generated: %s", res.get("report_id"))
    except Exception as e:
        logger.exception("Weekly pipeline failed: %s", e)

def _run_monthly():
    now = datetime.utcnow()
    start = now - timedelta(days=30)
    try:
        res = run_pipeline_for_period(start, now, "monthly")
        logger.info("Monthly ML report generated: %s", res.get("report_id"))
    except Exception as e:
        logger.exception("Monthly pipeline failed: %s", e)
