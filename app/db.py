# db.py
import mysql.connector
from mysql.connector import pooling
from pymongo import MongoClient
from app.config import *

# -----------------------------
# MYSQL CONNECTION POOL
# -----------------------------
pool = pooling.MySQLConnectionPool(
    pool_name="my_pool",
    pool_size=5,
    host=MYSQL_HOST,
    user=MYSQL_USER,
    password=MYSQL_PASSWORD,
    database=MYSQL_NAME,
)

def get_conn():
    return pool.get_connection()


# -----------------------------
# MONGO CONNECTION
# -----------------------------
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
