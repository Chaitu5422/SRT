# utils.py
from datetime import datetime, timedelta
from jose import jwt, JWTError
from fastapi import Header, HTTPException
from app.config import JWT_SECRET, JWT_ALGO, JWT_EXPIRE_MINUTES

# -------------------------------
# JWT CONFIG
# -------------------------------
SECRET_KEY = JWT_SECRET
ALGORITHM = JWT_ALGO
TOKEN_EXPIRE_MINUTES = JWT_EXPIRE_MINUTES

# -------------------------------
# CREATE TOKEN
# -------------------------------
def create_token(email: str):
    payload = {
        "email": email,
        "exp": datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRE_MINUTES)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


# -------------------------------
# VERIFY TOKEN
# -------------------------------
def verify_token(token: str):
    try:
        decoded = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return decoded
    except JWTError:
        return None


# -------------------------------
# AUTH MIDDLEWARE (DEPENDENCY)
# -------------------------------
def authorize(authorization: str = Header(None)):
    if authorization is None:
        raise HTTPException(401, "Authorization token missing")

    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid authorization header")

    token = authorization.split(" ")[1]
    decoded = verify_token(token)

    if decoded is None:
        raise HTTPException(401, "Invalid or expired token")

    return decoded   # contains email
