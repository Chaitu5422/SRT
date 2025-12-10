from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from app.auth import router as auth_router
from app.dashboard import router as dashboard_router
from app.reflection_mongo import router as reflection_router
from app.ai import router as ai_router
from app.timesheet import router as timesheet_router
from app.ml_routes import router as ml_router
from ml.reports_api import router as reports_router

load_dotenv()
app = FastAPI(title="MyReflection Login API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(auth_router)
app.include_router(dashboard_router)
app.include_router(reflection_router)
app.include_router(timesheet_router)
app.include_router(ai_router)
app.include_router(ml_router)          # /api/ml/*
app.include_router(reports_router)     # /api/reports/*
