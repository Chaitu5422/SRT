# app/timesheet_schema.py
from pydantic import BaseModel
from typing import List, Optional

class TaskItem(BaseModel):
    task: str
    hours: float

class TimeSheet(BaseModel):
    empID: str
    date: str
    project: str
    client: str
    tasks: List[TaskItem]

class ReflectionSubmit(BaseModel):
    email: str
    period: str  # daily / weekly / monthly

    # Daily
    selected_date: Optional[str] = None

    # Weekly
    week_start: Optional[str] = None
    week_end: Optional[str] = None

    # Monthly
    month: Optional[int] = None
    year: Optional[int] = None

    # Questions
    q1: str
    q2: str
    q3: str
