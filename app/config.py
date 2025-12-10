# app/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# ----------------------------------------
# 🔵 MYSQL CONFIG
# ----------------------------------------
MYSQL_HOST = os.getenv("DB_HOST")
MYSQL_USER = os.getenv("DB_USER")
MYSQL_PASSWORD = os.getenv("DB_PASS")
MYSQL_NAME = os.getenv("DB_NAME")

MYSQL_POOL_NAME = "myreflection_pool"
MYSQL_POOL_SIZE = 5

# ----------------------------------------
# 🟢 MONGO CONFIG
# ----------------------------------------
MONGO_URL = os.getenv("MONGO_URL", "mongodb://127.0.0.1:27017")
MONGO_DB_NAME = "myreflection"

# ----------------------------------------
# 🟣 JWT CONFIG
# ----------------------------------------
JWT_SECRET = os.getenv("JWT_SECRET", "SUPER_SECRET_KEY_CHANGE_THIS")
JWT_ALGO = "HS256"
JWT_EXPIRE_MINUTES = 60 * 24   # 1 day

# ----------------------------------------
# 🔥 OLLAMA / AI MODEL CONFIG
# ----------------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
AI_MODEL = os.getenv("AI_MODEL", "llama3")

# ----------------------------------------
# ✉ EMAIL CONFIG
# ----------------------------------------
EMAIL_SENDER = os.getenv("EMAIL_SENDER", "chaitanya9636@gmail.com")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "ocob hnok srsu sxly")
EMAIL_SMTP = "smtp.gmail.com"
EMAIL_PORT = 587
