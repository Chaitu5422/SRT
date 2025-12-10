# db.py
import os
from psycopg2 import pool
from pymongo import MongoClient

# -----------------------------
# POSTGRESQL CONNECTION POOL
# -----------------------------
# Read DATABASE_URL from environment (Render provides this)
DATABASE_URL = os.getenv("DATABASE_URL")

# Create a simple connection pool (psycopg2 pool)
# Note: psycopg2 pool is less feature-rich than MySQL's, but works well
pg_pool = pool.SimpleConnectionPool(
    1,  # minconn
    5,  # maxconn
    DATABASE_URL
)

def get_postgres_conn():
    return pg_pool.getconn()

def put_postgres_conn(conn):
    pg_pool.putconn(conn)


# -----------------------------
# MONGO CONNECTION (Atlas)
# -----------------------------
MONGO_URL = os.getenv("MONGO_URL")

def get_mongo_client():
    return MongoClient(MONGO_URL)

# ============================================================
# COMMON COLLECTION HELPERS
# ============================================================
def get_reflection_collection():
    client = get_mongo_client()
    db = client["myreflection"]
    return db["reflections"]

def get_timesheet_collection():
    client = get_mongo_client()
    db = client["myreflection"]
    return db["timesheets"]

def get_ml_reports_collection():
    client = get_mongo_client()
    db = client["myreflection"]
    return db["ml_reports"]
