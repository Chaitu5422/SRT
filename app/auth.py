from fastapi import APIRouter, HTTPException, Form, Query
from app.db import get_conn
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from app.utils import create_token
from app.utils import authorize
import random
from app.config import EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_SMTP, EMAIL_PORT

router = APIRouter(prefix="/api/auth", tags=["Authentication"])

# OTP memory store (no database)
OTP_STORE = {}  # {email: {"otp": "123456", "expires": datetime}}


# ============================================================
# 1️⃣ FORGOT PASSWORD → Send OTP (NO DB STORAGE)
# ============================================================
@router.post("/forgot_password")
def forgot_password(email: str = Form(...)):
    conn = get_conn()
    cur = conn.cursor(dictionary=True)

    # Fetch user record
    cur.execute("SELECT Name, EmailID FROM employees WHERE EmailID=%s", (email,))
    user = cur.fetchone()

    cur.close()
    conn.close()

    if not user:
        raise HTTPException(404, "Email not found")

    user_name = user.get("Name") or "there"

    # Generate OTP
    otp = str(random.randint(100000, 999999))

    # Store OTP temporarily
    OTP_STORE[email] = {
        "otp": otp,
        "expires_at": datetime.utcnow() + timedelta(minutes=10)
    }

    # ==========================
    # ✉ PROFESSIONAL EMAIL TEMPLATE (WITH NAME)
    # ==========================
    html_content = f"""
    <html>
    <body style="font-family: Arial, sans-serif; background-color: #f5f7fa; padding: 20px;">
        <div style="max-width: 500px; margin: auto; background: white; padding: 25px;
                    border-radius: 10px; box-shadow: 0 3px 12px rgba(0,0,0,0.1);">

            <h2 style="color: #333; text-align: center;">🔐 Password Reset Request</h2>

            <p style="font-size: 15px; color: #555;">
                Hello <strong>{user_name}</strong>,
            </p>

            <p style="font-size: 15px; color: #555;">
                We received a request to reset your password for your 
                <strong>Reflection Portal</strong> account.
            </p>

            <div style="text-align: center; margin: 25px 0;">
                <p style="font-size: 14px; color: #555;">Your One-Time Password (OTP) is:</p>

                <div style="font-size: 28px; font-weight: bold; color: #2a7ae2;
                            background: #eef4ff; padding: 12px 0; border-radius: 8px;">
                    {otp}
                </div>

                <p style="margin-top: 10px; font-size: 13px; color: #888;">
                    This OTP is valid for <strong>10 minutes</strong>.
                </p>
            </div>

            <p style="font-size: 14px; color: #666;">
                If you did not request a password reset, please ignore this email.
            </p>

            <br/>

            <p style="font-size: 14px; color: #666;">Regards,<br/>
                <strong>Reflection Portal Team</strong></p>

            <hr style="border: none; border-top: 1px solid #eee; margin: 25px 0;"/>

            <p style="font-size: 12px; color: #aaa; text-align: center;">
                This is an automated email. Please do not reply.
            </p>
        </div>
    </body>
    </html>
    """

    # Send OTP
    try:
        sender_email = EMAIL_SENDER
        sender_password = EMAIL_PASSWORD

        msg = MIMEText(html_content, "html")
        msg["Subject"] = "Reflection Portal - Password Reset OTP"
        msg["From"] = sender_email
        msg["To"] = email

        server = smtplib.SMTP(EMAIL_SMTP, EMAIL_PORT)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, [email], msg.as_string())
        server.quit()

    except Exception as e:
        print("Error sending email:", e)
        raise HTTPException(500, "Failed to send OTP email")

    return {"status": "success", "message": "OTP sent to your email"}


# ============================================================
# 2️⃣ VERIFY OTP
# ============================================================
@router.post("/verify_otp")
def verify_otp(email: str = Form(...), otp: str = Form(...)):
    entry = OTP_STORE.get(email)

    if not entry:
        raise HTTPException(400, "No OTP generated for this email")

    if datetime.utcnow() > entry["expires_at"]:
        del OTP_STORE[email]
        raise HTTPException(400, "OTP expired")

    if otp != entry["otp"]:
        raise HTTPException(400, "Invalid OTP")

    return {"status": "success", "message": "OTP verified"}


# ============================================================
# 3️⃣ RESET PASSWORD
# ============================================================
@router.post("/reset_password")
def reset_password(email: str = Form(...), otp: str = Form(...), new_password: str = Form(...)):
    entry = OTP_STORE.get(email)

    if not entry:
        raise HTTPException(400, "No OTP generated for this email")

    if datetime.utcnow() > entry["expires_at"]:
        del OTP_STORE[email]
        raise HTTPException(400, "OTP expired")

    if otp != entry["otp"]:
        raise HTTPException(400, "Invalid OTP")

    # OTP success → update password
    conn = get_conn()
    cur = conn.cursor(dictionary=True)

    cur.execute("UPDATE employees SET Password=%s WHERE EmailID=%s", (new_password, email))
    conn.commit()

    cur.close()
    conn.close()

    # Remove OTP after use
    del OTP_STORE[email]

    return {"status": "success", "message": "Password reset successfully"}


# ============================================================
# 4️⃣ LOGIN
# ============================================================
@router.post("/login")
def login(email: str = Form(...), password: str = Form(...)):
    conn = get_conn()
    cur = conn.cursor(dictionary=True)

    cur.execute("SELECT * FROM employees WHERE EmailID=%s", (email,))
    user = cur.fetchone()

    cur.close()
    conn.close()

    if not user:
        raise HTTPException(401, "User not found")

    if password.strip() != str(user["Password"]).strip():
        raise HTTPException(401, "Invalid credentials")

    token = create_token(email)

    return {
        "status": "success",
        "message": "Login successful",
        "token": token,
        "user": user["Name"],
        "empID": user["empID"],
        "designation": user["Designation"],
        "location": user["Location"]
    }



# ============================================================
# 5️⃣ HELPER: ROLE OPTIONS (for dropdown)
#    RoleID mapping used everywhere:
#    1 = SBUHead, 2 = Manager, 3 = Employee
# ============================================================
@router.get("/roles")
def get_roles():
    return {
        "status": "success",
        "roles": [
            {"id": 3, "name": "Employee"},
            {"id": 2, "name": "Manager"},
            {"id": 1, "name": "SBUHead"},
        ],
    }


# ============================================================
# 6️⃣ HELPER: Reporting Managers Dropdown
#     Show ONLY people with Designation containing 'Manager'
# ============================================================
@router.get("/reporting_managers")
def get_reporting_managers():
    conn = get_conn()
    cur = conn.cursor(dictionary=True)

    cur.execute("""
        SELECT ID, empID, Name, EmailID, Designation
        FROM employees
        WHERE LOWER(Designation) LIKE '%manager%'
        ORDER BY Name
    """)

    rows = cur.fetchall()
    cur.close()
    conn.close()

    return {
        "status": "success",
        "managers": rows,
    }


# ============================================================
# 7️⃣ HELPER: SBUHeads Dropdown
#     Only show Designation = 'SBU Head' or 'MD'
# ============================================================
@router.get("/sbu_heads")
def get_sbu_heads():
    conn = get_conn()
    cur = conn.cursor(dictionary=True)

    cur.execute("""
        SELECT ID, empID, Name, EmailID, Designation
        FROM employees
        WHERE LOWER(Designation) IN ('sbu head', 'md')
        ORDER BY Name
    """)

    rows = cur.fetchall()
    cur.close()
    conn.close()

    return {
        "status": "success",
        "sbuHeads": rows,
    }


# ============================================================
# 8️⃣ HELPER: Companies
# ============================================================
@router.get("/companies")
def get_companies():
    conn = get_conn()
    cur = conn.cursor(dictionary=True)

    try:
        cur.execute("SELECT CompanyID, CompanyName FROM companies ORDER BY CompanyName")
        rows = cur.fetchall()
    except Exception:
        rows = []

    cur.close()
    conn.close()

    return {"status": "success", "companies": rows}


# ============================================================
# 9️⃣ HELPER: Departments
# ============================================================
@router.get("/departments")
def get_departments():
    conn = get_conn()
    cur = conn.cursor(dictionary=True)

    try:
        cur.execute("SELECT DepartmentID, DepartmentName FROM departments ORDER BY DepartmentName")
        rows = cur.fetchall()
    except Exception:
        rows = []

    cur.close()
    conn.close()

    return {"status": "success", "departments": rows}


# ============================================================
# 🔟 USER REGISTRATION  – with strict RM / SBUHead validation
# ============================================================
@router.post("/register")
def register_user(
    empID: str = Form(...),
    name: str = Form(...),
    designation: str = Form(...),
    location: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    roleName: str = Form(...),
    reportingManagerName: str = Form(None),
    companyID: str = Form(None),
    departmentID: str = Form(None),
    sbuHeadName: str = Form(None),
    isActive: str = Form("true"),
):
    conn = get_conn()
    cur = conn.cursor(dictionary=True)

    # Trim inputs
    empID = empID.strip()
    email = email.strip()
    name = name.strip()
    designation = designation.strip()
    location = location.strip()
    roleName = roleName.strip().lower()

    # --- Unique Check ---
    cur.execute("SELECT ID FROM employees WHERE empID=%s OR EmailID=%s", (empID, email))
    if cur.fetchone():
        cur.close(); conn.close()
        raise HTTPException(400, "Employee with this empID or Email already exists")

    # --- Role Mapping ---
    role_map = {"employee": 3, "manager": 2, "sbuhead": 1, "sbu head": 1, "sbu_head": 1}
    role_id = role_map.get(roleName)
    if not role_id:
        cur.close(); conn.close()
        raise HTTPException(400, "Invalid role name")

    # --- Role Based Validation ---
    rm_name = (reportingManagerName or "").strip()
    sh_name = (sbuHeadName or "").strip()

    # Employee → RM mandatory, SBU optional
    if role_id == 3 and not rm_name:
        raise HTTPException(400, "Reporting Manager is required for employees")

    # Manager → RM=SBU only (SBU is reporting manager)
    if role_id == 2:
        if not sh_name:
            raise HTTPException(400, "SBU Head is required for Managers")
        rm_name = sh_name  # Auto bind

    # SBU Head → RM=SBU only (self + other SBU allowed)
    if role_id == 1:
        if not sh_name:
            raise HTTPException(400, "SBU Head selection is required")
        rm_name = sh_name  # Auto bind

    # --- Validate Reporting Manager ---
    reporting_manager_id = None
    if rm_name:
        cur.execute("""
            SELECT ID FROM employees 
            WHERE LOWER(Name)=LOWER(%s)
        """, (rm_name,))
        rm = cur.fetchone()
        if not rm:
            raise HTTPException(400, "Selected Reporting Manager not found")
        reporting_manager_id = rm["ID"]

    # --- Validate SBU Head ---
    sbu_head_id = None
    if sh_name:
        cur.execute("""
            SELECT ID FROM employees 
            WHERE LOWER(Name)=LOWER(%s)
              AND LOWER(Designation) IN ('sbu head','md')
        """, (sh_name,))
        sh = cur.fetchone()
        if not sh:
            raise HTTPException(400, "Invalid SBU Head selection")
        sbu_head_id = sh["ID"]

    # For Manager/SBU: ensure BOTH are same ID
    if role_id in (1, 2):
        if reporting_manager_id != sbu_head_id:
            raise HTTPException(
                400,
                "Reporting Manager and SBU Head must be the same person for Managers/SBU Heads"
            )

    # --- Validate Optional INT fields ---
    def normalize(v):
        if not v or str(v).strip().lower() in ("null", "none", ""):
            return None
        try: return int(v)
        except: return None

    company_id_val = normalize(companyID)
    department_id_val = normalize(departmentID)
    is_active_val = 1 if isActive.lower() in ("true","1","yes","on") else 0

    # --- Insert ---
    cur.execute("""
        INSERT INTO employees
        (empID, Name, Designation, Location, EmailID, Password,
         RoleID, ReportingManagerID, CompanyID, DepartmentID, SBUHeadID, IsActive)
        VALUES (%s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s)
    """, (
        empID, name, designation, location, email, password,
        role_id, reporting_manager_id,
        company_id_val, department_id_val,
        sbu_head_id, is_active_val
    ))

    conn.commit()
    new_id = cur.lastrowid
    cur.close(); conn.close()

    return {
        "status": "success",
        "message": "Registration successful",
        "ID": new_id,
        "empID": empID,
        "email": email
    }
