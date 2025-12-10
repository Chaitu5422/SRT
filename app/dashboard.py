from fastapi import APIRouter, HTTPException, Query, Depends
from app.db import get_conn
from app.utils import authorize

router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])

# ==========================================================
# GET EMPLOYEE DETAILS
# Adds: canViewTeam = True/False
# ==========================================================
@router.get("/employee")
def get_employee(email: str = Query(...)):
    conn = get_conn()
    cur = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT ID, empID, Name, Designation, Location, EmailID, RoleID, ReportingManagerID 
        FROM employees WHERE EmailID=%s
    """, (email,))
    emp = cur.fetchone()
    cur.close()
    conn.close()

    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    # Team visibility -> only RoleID 1 and 2
    can_view_team = emp["RoleID"] in (1, 2)

    return {
        "status": "success",
        "employee": emp,
        "canViewTeam": can_view_team   # 👈 Required for frontend UI hiding
    }


# ==========================================================
# VIEW TEAM (Dashboard Top Section)
# Employee (RoleID = 3) → BLOCKED
# ==========================================================
@router.get("/view_team")
def view_team(email: str = Query(...)):
    conn = get_conn()
    cur = conn.cursor(dictionary=True)

    # Logged-in user
    cur.execute("""
        SELECT ID, Name, RoleID, ReportingManagerID 
        FROM employees WHERE EmailID=%s
    """, (email,))
    logged_user = cur.fetchone()

    if not logged_user:
        cur.close()
        conn.close()
        raise HTTPException(status_code=404, detail="User not found")

    role = logged_user["RoleID"]
    user_id = logged_user["ID"]

    # ❌ Employee should NOT see team view
    if role == 3:
        cur.close()
        conn.close()
        return {
            "status": "forbidden",
            "message": "Team view is only available for Managers and SBU Heads."
        }

    records = []

    # -------------------------------------------
    # Case 1: SBU Head (Role 1)
    # -------------------------------------------
    if role == 1:
        # Self
        cur.execute("""
            SELECT ID, empID, Name, Designation, Location, EmailID, RoleID, ReportingManagerID
            FROM employees WHERE ID=%s
        """, (user_id,))
        records.extend(cur.fetchall())

        # Managers under SBU
        cur.execute("""
            SELECT ID, empID, Name, Designation, Location, EmailID, RoleID, ReportingManagerID
            FROM employees
            WHERE ReportingManagerID=%s AND RoleID=2
        """, (user_id,))
        managers = cur.fetchall()
        records.extend(managers)

        # Members under each manager
        manager_ids = [m["ID"] for m in managers]
        if manager_ids:
            placeholders = ",".join(["%s"] * len(manager_ids))
            cur.execute(f"""
                SELECT ID, empID, Name, Designation, Location, EmailID, RoleID, ReportingManagerID
                FROM employees
                WHERE ReportingManagerID IN ({placeholders})
            """, tuple(manager_ids))
            records.extend(cur.fetchall())

    # -------------------------------------------
    # Case 2: Manager (Role 2)
    # -------------------------------------------
    elif role == 2:
        cur.execute("""
            SELECT ID, empID, Name, Designation, Location, EmailID, RoleID, ReportingManagerID
            FROM employees
            WHERE ID=%s OR ReportingManagerID=%s
        """, (user_id, user_id))
        records = cur.fetchall()

    cur.close()
    conn.close()

    return {"status": "success", "viewer_role": role, "records": records}


# ==========================================================
# TEAM MEMBERS (3-Level Dropdown Logic)
# RoleID = 3 → BLOCKED
# ==========================================================
@router.get("/team_members")
def get_team_members(email: str = Query(...), manager_email: str = Query(None), sub_manager_email: str = Query(None)):

    conn = get_conn()
    cur = conn.cursor(dictionary=True)

    # Logged-in user
    cur.execute("SELECT ID, RoleID, Name, EmailID FROM employees WHERE EmailID=%s", (email,))
    user = cur.fetchone()

    if not user:
        cur.close()
        conn.close()
        raise HTTPException(status_code=404, detail="User not found")

    user_id = user["ID"]
    user_role = user["RoleID"]

    # ❌ Employee (Role 3) cannot access team members
    if user_role == 3:
        cur.close()
        conn.close()
        return {
            "status": "forbidden",
            "message": "Team members view is only available for Managers and SBU Heads."
        }

    # =====================================================
    # CASE 1 — SBU HEAD
    # =====================================================
    if user_role == 1:

        # FIRST DROPDOWN
        if not manager_email:
            cur.execute("""
                SELECT ID, Name, EmailID, RoleID 
                FROM employees 
                WHERE ReportingManagerID=%s
            """, (user_id,))
            team = cur.fetchall()

            cur.close()
            conn.close()
            return {"status": "success", "level": 1, "team": team}

        # MANAGER SELECTED
        cur.execute("SELECT ID, Name, EmailID, RoleID FROM employees WHERE EmailID=%s", (manager_email,))
        selected_person = cur.fetchone()

        if not selected_person:
            cur.close()
            conn.close()
            raise HTTPException(404, "Selected user not found")

        selected_id = selected_person["ID"]

        # If SBU selected → list managers under SBU
        if selected_person["RoleID"] == 1:
            cur.execute("""
                SELECT ID, Name, EmailID, RoleID
                FROM employees
                WHERE ReportingManagerID=%s AND RoleID IN (2,3)
            """, (selected_id,))
            managers = cur.fetchall()

            cur.close()
            conn.close()
            return {"status": "success", "level": 2, "selected": selected_person, "team": managers}

        # Otherwise → get manager's team
        cur.execute("""
            SELECT Name, EmailID, RoleID 
            FROM employees 
            WHERE ReportingManagerID=%s
        """, (selected_id,))
        members = cur.fetchall()

        result = [{
            "Name": selected_person["Name"],
            "EmailID": selected_person["EmailID"],
            "RoleID": selected_person["RoleID"]
        }] + members

        cur.close()
        conn.close()
        return {"status": "success", "level": 3, "team": result}

    # =====================================================
    # CASE 2 — MANAGER
    # =====================================================
    elif user_role == 2:
        cur.execute("SELECT Name, EmailID, RoleID FROM employees WHERE ReportingManagerID=%s", (user_id,))
        team = cur.fetchall()

        cur.close()
        conn.close()

        return {
            "status": "success",
            "level": 3,
            "team": [{
                "Name": user["Name"],
                "EmailID": user["EmailID"],
                "RoleID": user_role
            }] + team
        }


# ==========================================================
# MANAGER TEAM VIEW (Extra API)
# RoleID 3 → BLOCKED
# ==========================================================
@router.get("/manager_team")
def get_manager_team(email: str = Query(...)):
    conn = get_conn()
    cur = conn.cursor(dictionary=True)

    cur.execute("SELECT ID, Name, RoleID, EmailID FROM employees WHERE EmailID=%s", (email,))
    user = cur.fetchone()

    if not user:
        cur.close()
        conn.close()
        raise HTTPException(404, "User not found")

    # ❌ Employees cannot use this API
    if user["RoleID"] == 3:
        cur.close()
        conn.close()
        return {
            "status": "forbidden",
            "message": "Only Managers and SBU Heads can view manager team."
        }

    user_id = user["ID"]

    # Detect if team exists
    cur.execute("SELECT COUNT(*) AS cnt FROM employees WHERE ReportingManagerID=%s", (user_id,))
    team_count = cur.fetchone()["cnt"]

    if team_count == 0:
        cur.close()
        conn.close()
        return {"status": "success", "team": []}

    team_list = [{
        "Name": f"{user['Name']} (Manager)",
        "EmailID": user["EmailID"],
        "RoleID": user["RoleID"]
    }]

    cur.execute("SELECT Name, EmailID, RoleID FROM employees WHERE ReportingManagerID=%s", (user_id,))
    team_list += cur.fetchall()

    cur.close()
    conn.close()

    return {"status": "success", "team": team_list}
