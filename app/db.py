# app/db.py
import os
from psycopg2 import pool
from pymongo import MongoClient

# -----------------------------
# POSTGRESQL CONNECTION POOL
# -----------------------------
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")

pg_pool = pool.SimpleConnectionPool(1, 5, DATABASE_URL)

def get_postgres_conn():
    return pg_pool.getconn()

def put_postgres_conn(conn):
    pg_pool.putconn(conn)

# -----------------------------
# MONGO CONNECTION
# -----------------------------
MONGO_URL = os.getenv("MONGO_URL")
if not MONGO_URL:
    raise ValueError("MONGO_URL environment variable is required")

def get_mongo_client():
    return MongoClient(MONGO_URL)

# -----------------------------
# COLLECTION HELPERS
# -----------------------------
def get_reflection_collection():
    client = get_mongo_client()
    return client["myreflection"]["reflections"]

def get_timesheet_collection():
    client = get_mongo_client()
    return client["myreflection"]["timesheets"]

def get_ml_reports_collection():
    client = get_mongo_client()
    return client["myreflection"]["ml_reports"]
