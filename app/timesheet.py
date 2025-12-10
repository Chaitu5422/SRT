from fastapi import APIRouter, HTTPException, Form, Query, Depends
from app.db import get_timesheet_collection
from app.db import get_conn
from fastapi import Body
from app.utils import authorize
from datetime import datetime, date as date_cls, timedelta
import json
from typing import Optional, Dict, Any, List

router = APIRouter(prefix="/api/timesheet", tags=["Timesheet"])

# ---------------- COMMON DATE HELPERS ----------------
def parse_ymd(date_str: str) -> date_cls:
    """
    Safely parse date coming from frontend.
    Supports:
        - 'YYYY-MM-DD'
        - 'YYYY-MM-DDTHH:MM:SS'
        - 'YYYY-MM-DDTHH:MM:SSZ'
    """
    try:
        # Remove timezone Z if present and split time part
        date_only = date_str.split("T")[0]
        return datetime.strptime(date_only, "%Y-%m-%d").date()
    except Exception:
        raise HTTPException(400, "Invalid date format. Use YYYY-MM-DD")


def day_start_end(d: date_cls):
    """
    Returns start (00:00) and end (23:59:59) datetime for a given date.
    Useful to be robust even if old docs have different times.
    """
    start = datetime.combine(d, datetime.min.time())
    end = datetime.combine(d, datetime.max.time())
    return start, end


# ---------------- SUBMIT TIMESHEET ----------------
@router.post("/submit")
def submit_timesheet(
    #payload=Depends(authorize),
    empID: str = Form(...),
    date: str = Form(...),       # Expecting YYYY-MM-DD (but also handles ISO)
    tasks: str = Form(...),      # JSON string (list of tasks)
    totalHours: float = Form(...)
):
    # ---------- 1) Validate employee ----------
    conn = get_conn()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT empID, Name, Designation FROM employees WHERE empID=%s",
            (empID,)
        )
        emp = cur.fetchone()
    finally:
        cur.close()
        conn.close()

    if not emp:
        raise HTTPException(404, "Employee not found")

    # ---------- 2) Normalize & validate date ----------
    selected_date_obj = parse_ymd(date)

    # Prevent future timesheet
    if selected_date_obj > date_cls.today():
        raise HTTPException(400, "Timesheet cannot be submitted for future dates.")

    # We'll use canonical midnight datetime for this date
    selected_dt = datetime.combine(selected_date_obj, datetime.min.time())

    # ---------- 3) Parse tasks JSON ----------
    # Expecting something like:
    # [
    #   {"project": "AI Tracker", "task": "API work", "hours": 4},
    #   {"project": "Portal", "task": "UI Fix", "hours": 3.5}
    # ]
    try:
        tasks_json = json.loads(tasks)
        if not isinstance(tasks_json, list):
            raise ValueError("tasks must be a list")
    except Exception:
        raise HTTPException(400, "Invalid tasks JSON format. Expecting a JSON array list.")

    # ---------- 4) Check duplicate for that date ----------
    col = get_timesheet_collection()

    # More robust duplicate check: any doc on that day for this emp
    day_start, day_end = day_start_end(selected_date_obj)

    duplicate = col.find_one({
        "empID": empID,
        "SelectedDate": {"$gte": day_start, "$lte": day_end}
    })

    if duplicate:
        raise HTTPException(400, "Timesheet already submitted for this date")

    # ---------- 5) Prepare document ----------
    doc = {
        "empID": emp["empID"],
        "Name": emp["Name"],
        "Designation": emp["Designation"],

        # Same style as reflections:
        "SelectedDate": selected_dt,                              # canonical datetime
        "DateSubmitted": date_cls.today().strftime("%Y-%m-%d"),   # simple date string
        "CreatedAt": datetime.utcnow().replace(microsecond=0),    # ISO clean UTC

        "tasks": tasks_json,
        "totalHours": totalHours
    }

    col.insert_one(doc)
    return {"status": "success", "message": "Timesheet submitted successfully ✔️"}


# ---------------- HISTORY FILTER ----------------
@router.get("/history")
def timesheet_history(
    empID: str = Query(...),
    project: Optional[str] = Query(None),

    date: Optional[str] = Query(None),           # YYYY-MM-DD
    fromDate: Optional[str] = Query(None),       # YYYY-MM-DD
    toDate: Optional[str] = Query(None),         # YYYY-MM-DD

    week: Optional[str] = Query(None),           # 'this', 'last', or 'YYYY-W05'
    month: Optional[str] = Query(None)           # YYYY-MM
):
    col = get_timesheet_collection()
    query: Dict[str, Any] = {"empID": empID}

    # ---------- Project filter ----------
    # tasks is a list, so use $elemMatch for project based search (case-insensitive)
    if project:
        query["tasks"] = {
            "$elemMatch": {
                "project": {"$regex": project, "$options": "i"}
            }
        }

    # ---------- Date filter (single day) ----------
    if date:
        d = parse_ymd(date)
        start, end = day_start_end(d)
        query["SelectedDate"] = {"$gte": start, "$lte": end}

    # ---------- Range filter ----------
    # Only apply if at least one is present and date not already specified
    if (fromDate or toDate) and "SelectedDate" not in query:
        range_query: Dict[str, Any] = {}
        if fromDate:
            fd = parse_ymd(fromDate)
            range_query["$gte"] = datetime.combine(fd, datetime.min.time())
        if toDate:
            td = parse_ymd(toDate)
            range_query["$lte"] = datetime.combine(td, datetime.max.time())

        if range_query:
            query["SelectedDate"] = range_query

    # ---------- Week filter ----------
    # Priority: if week is given, ignore simple date & range (only week applied)
    if week:
        today = date_cls.today()

        # Overwrite any previous SelectedDate filter
        if week == "this":
            start = today - timedelta(days=today.weekday())  # Monday of this week
        elif week == "last":
            # Monday of last week
            start = today - timedelta(days=today.weekday() + 7)
        else:
            # Format: YYYY-W05
            try:
                year_str, week_str = week.split("-W")
                year = int(year_str)
                week_no = int(week_str)
                # Monday = 1, Sunday = 7
                start = date_cls.fromisocalendar(year, week_no, 1)
            except Exception:
                raise HTTPException(400, "Invalid week format. Use 'this', 'last' or 'YYYY-Www'")

        end = start + timedelta(days=6)
        query["SelectedDate"] = {
            "$gte": datetime.combine(start, datetime.min.time()),
            "$lte": datetime.combine(end, datetime.max.time())
        }

    # ---------- Month filter ----------
    # If month is passed, it overrides day / range / week filter
    if month:
        try:
            yyyy, mm = map(int, month.split("-"))
            start = date_cls(yyyy, mm, 1)
        except Exception:
            raise HTTPException(400, "Invalid month format. Use YYYY-MM")

        # Next month calculation
        if mm == 12:
            next_month = date_cls(yyyy + 1, 1, 1)
        else:
            next_month = date_cls(yyyy, mm + 1, 1)

        end = next_month - timedelta(days=1)

        query["SelectedDate"] = {
            "$gte": datetime.combine(start, datetime.min.time()),
            "$lte": datetime.combine(end, datetime.max.time())
        }

    # ---------- Execute query ----------
    results: List[Dict[str, Any]] = list(
        col.find(query, {"_id": 0}).sort("SelectedDate", -1)
    )

    return {
        "status": "success",
        "count": len(results),
        "records": results
    }


# ---------------- TEAM VIEW ----------------
@router.get("/team")
def team_timesheet_view(email: str = Query(...)):
    """
    - RoleID = 3 -> Employee (no team access)
    - RoleID = 2 -> Manager (self + direct reports)
    - RoleID = 1 -> SBU Head (self + managers + their team)
    """
    # ---------- Get logged-in user ----------
    conn = get_conn()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT ID, empID, Name, RoleID FROM employees WHERE EmailID=%s",
            (email,)
        )
        user = cur.fetchone()
    finally:
        cur.close()
        conn.close()

    if not user:
        raise HTTPException(404, "User not found")

    role = user["RoleID"]
    user_id = user["ID"]
    col = get_timesheet_collection()

    # ---------- Employee – forbidden ----------
    if role == 3:
        return {
            "status": "forbidden",
            "message": "Only Managers and SBU Heads can view team timesheets."
        }

    # ---------- SBU Head ----------
    if role == 1:
        # 1. Get direct managers under this SBU
        conn = get_conn()
        try:
            cur = conn.cursor(dictionary=True)
            cur.execute(
                "SELECT empID, ID FROM employees WHERE ReportingManagerID=%s AND RoleID=2",
                (user_id,)
            )
            managers = cur.fetchall()
        finally:
            cur.close()
            conn.close()

        ids = [user["empID"]]  # SBU self
        mgr_ids = []

        if managers:
            mgr_ids = [x["ID"] for x in managers]
            ids += [x["empID"] for x in managers]

        # 2. Get employees under these managers
        if mgr_ids:
            placeholders = ",".join(["%s"] * len(mgr_ids))
            conn = get_conn()
            try:
                cur = conn.cursor(dictionary=True)
                cur.execute(
                    f"SELECT empID FROM employees WHERE ReportingManagerID IN ({placeholders})",
                    tuple(mgr_ids)
                )
                members = cur.fetchall()
            finally:
                cur.close()
                conn.close()

            ids += [m["empID"] for m in members]

        recs = list(col.find({"empID": {"$in": ids}}, {"_id": 0}).sort("SelectedDate", -1))
        return {"status": "success", "role": "SBU", "records": recs}

    # ---------- Manager ----------
    if role == 2:
        conn = get_conn()
        try:
            cur = conn.cursor(dictionary=True)
            cur.execute(
                "SELECT empID FROM employees WHERE ID=%s OR ReportingManagerID=%s",
                (user_id, user_id)
            )
            rows = cur.fetchall()
        finally:
            cur.close()
            conn.close()

        ids = [r["empID"] for r in rows]
        recs = list(col.find({"empID": {"$in": ids}}, {"_id": 0}).sort("SelectedDate", -1))
        return {"status": "success", "role": "Manager", "records": recs}

    # Fallback (if some other role)
    return {"status": "forbidden", "message": "Role not allowed to view team timesheets."}
# ---------------------------------------------------------
# ✏️ UPDATE TIMESHEET
# ---------------------------------------------------------
@router.patch("/update")
def update_timesheet(
    empID: str = Form(...),
    date: str = Form(...),
    tasks: str = Form(None),
    totalHours: float = Form(None)
):
    col = get_timesheet_collection()
    
    # Parse date
    selected = parse_ymd(date)
    start, end = day_start_end(selected)

    old_doc = col.find_one({
        "empID": empID,
        "SelectedDate": {"$gte": start, "$lte": end}
    })

    if not old_doc:
        raise HTTPException(404, "No timesheet found for this date")

    update_data = {}

    # update tasks if provided
    if tasks:
        try:
            update_data["tasks"] = json.loads(tasks)
        except:
            raise HTTPException(400, "Invalid tasks JSON format")

    # update total hours
    if totalHours is not None:
        update_data["totalHours"] = totalHours

    if not update_data:
        return {"status": "ignored", "message": "Nothing to update"}

    col.update_one(
        {"_id": old_doc["_id"]},
        {"$set": update_data}
    )

    return {"status": "success", "message": "Timesheet updated successfully"}


# ---------------------------------------------------------
# 🗑 DELETE TIMESHEET
# ---------------------------------------------------------
@router.delete("/delete")
def delete_timesheet(empID: str = Query(...), date: str = Query(...)):
    col = get_timesheet_collection()

    d = parse_ymd(date)
    start, end = day_start_end(d)

    result = col.delete_one({
        "empID": empID,
        "SelectedDate": {"$gte": start, "$lte": end}
    })

    if result.deleted_count == 0:
        raise HTTPException(404, "No timesheet found to delete")

    return {"status": "success", "message": "Timesheet deleted successfully"}


# ---------------- TEAM HIERARCHY VIEW + FILTERS ----------------
# ---------------- TIMESHEET HIERARCHY VIEW (MATCHES REFLECTION LOGIC) ----------------
@router.get("/team_hierarchy")
def team_timesheet_hierarchy(
    email: str = Query(...),

    project: Optional[str] = Query(None),
    date: Optional[str] = Query(None),
    fromDate: Optional[str] = Query(None),
    toDate: Optional[str] = Query(None),
    week: Optional[str] = Query(None),
    month: Optional[str] = Query(None)
):
    # 1️⃣ Get logged-in user details
    conn = get_conn()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT ID, empID, RoleID, ReportingManagerID FROM employees WHERE EmailID=%s", (email,))
    user = cur.fetchone()
    cur.close(); conn.close()

    if not user:
        raise HTTPException(404, "User not found")

    user_id = user["ID"]
    role = user["RoleID"]
    user_empid = user["empID"]

    col = get_timesheet_collection()

    # 2️⃣ Build team hierarchy (same as reflection)
    emp_ids = []

    # --------------------- SBU HEAD ---------------------
    if role == 1:
        conn = get_conn()
        cur = conn.cursor(dictionary=True)

        # Get all managers under SBU
        cur.execute("SELECT ID, empID FROM employees WHERE ReportingManagerID=%s AND RoleID=2", (user_id,))
        managers = cur.fetchall()

        manager_ids = [m["ID"] for m in managers]
        emp_ids = [user_empid] + [m["empID"] for m in managers]

        # Get all employees under each manager
        if manager_ids:
            placeholders = ",".join(["%s"] * len(manager_ids))
            cur.execute(
                f"SELECT empID FROM employees WHERE ReportingManagerID IN ({placeholders})",
                tuple(manager_ids)
            )
            members = cur.fetchall()
            emp_ids += [m["empID"] for m in members]

        cur.close(); conn.close()

    # --------------------- MANAGER ---------------------
    elif role == 2:
        conn = get_conn()
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT empID FROM employees WHERE ID=%s OR ReportingManagerID=%s", (user_id, user_id))
        rows = cur.fetchall()
        emp_ids = [r["empID"] for r in rows]
        cur.close(); conn.close()

    # --------------------- EMPLOYEE ---------------------
    else:
        emp_ids = [user_empid]

    # Base filter
    query: Dict[str, Any] = {"empID": {"$in": emp_ids}}

    # 3️⃣ Apply filters (daily / weekly / monthly / project)
    if project:
        query["tasks"] = {
            "$elemMatch": {"project": {"$regex": project, "$options": "i"}}
        }

    # Single day
    if date:
        d = parse_ymd(date)
        start, end = day_start_end(d)
        query["SelectedDate"] = {"$gte": start, "$lte": end}

    # Date range
    if (fromDate or toDate) and "SelectedDate" not in query:
        r = {}
        if fromDate:
            fd = parse_ymd(fromDate)
            r["$gte"] = datetime.combine(fd, datetime.min.time())
        if toDate:
            td = parse_ymd(toDate)
            r["$lte"] = datetime.combine(td, datetime.max.time())
        query["SelectedDate"] = r

    # Week filter
    if week:
        year_str, week_str = week.split("-W")
        start = date_cls.fromisocalendar(int(year_str), int(week_str), 1)
        end = start + timedelta(days=6)
        query["SelectedDate"] = {
            "$gte": datetime.combine(start, datetime.min.time()),
            "$lte": datetime.combine(end, datetime.max.time())
        }

    # Month filter
    if month:
        yyyy, mm = map(int, month.split("-"))
        start = date_cls(yyyy, mm, 1)
        end = (date_cls(yyyy+1,1,1) if mm == 12 else date_cls(yyyy, mm+1,1)) - timedelta(days=1)
        query["SelectedDate"] = {
            "$gte": datetime.combine(start, datetime.min.time()),
            "$lte": datetime.combine(end, datetime.max.time())
        }

    # 4️⃣ Execute
    results = list(col.find(query, {"_id": 0}).sort("SelectedDate", -1))

    return {
        "status": "success",
        "count": len(results),
        "records": results
    }