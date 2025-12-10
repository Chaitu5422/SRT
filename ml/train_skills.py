# backend/ml/train_skills.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
import joblib
import os

# ---------------------------------------
# TRAINING DATA (50+ balanced examples)
# ---------------------------------------

TRAIN_DATA = [

    # ---------------- PYTHON ----------------
    ("I wrote a Python script to automate tasks", ["python"]),
    ("Working on Python backend API with FastAPI", ["python", "api_development"]),
    ("Developed a Python data processing tool", ["python"]),
    ("Optimized Python code for better speed", ["python"]),
    ("Created a Python backend service for employees", ["python"]),
    ("Built a Python API system for data exchange", ["python", "api_development"]),
    ("Improved Python backend logic", ["python"]),
    ("Used Python to clean and process data", ["python"]),

    # ---------------- SQL ----------------
    ("I fixed SQL queries and optimized the database", ["sql"]),
    ("Created tables and wrote SQL joins", ["sql"]),
    ("Performed SQL indexing and query optimization", ["sql"]),
    ("Wrote SQL scripts for data migration", ["sql"]),
    ("Improved SQL query performance", ["sql"]),

    # ---------------- API DEVELOPMENT ----------------
    ("Designed and implemented REST API endpoints", ["api_development"]),
    ("Created CRUD API using FastAPI", ["api_development"]),
    ("Optimized API response time", ["api_development"]),
    ("Integrated backend API with user dashboard", ["api_development"]),
    ("Worked on API bug fixes and improvements", ["api_development"]),
    ("Wrote API logic for user management", ["api_development"]),

    # ---------------- TESTING ----------------
    ("Tested the API and fixed bugs", ["testing", "api_development"]),
    ("Wrote unit test cases for the module", ["testing"]),
    ("Debugged production issues and logs", ["testing"]),
    ("Performed regression testing for the backend", ["testing"]),
    ("Wrote unit tests for backend API", ["testing", "api_development"]),
    ("Fixed failing test cases", ["testing"]),

    # ---------------- COMMUNICATION ----------------
    ("Had a client meeting and discussed requirements", ["communication"]),
    ("Presented my work to the team", ["communication"]),
    ("Conducted a knowledge-sharing session", ["communication"]),
    ("Communicated updates to the project team", ["communication"]),
    ("Collaborated with teammates on project goals", ["communication"]),

    # ---------------- LEADERSHIP ----------------
    ("Guided junior team members", ["leadership"]),
    ("Planned sprint tasks and assigned responsibilities", ["leadership"]),
    ("Mentored new joiners on the project", ["leadership"]),
    ("Handled team coordination and task delegation", ["leadership"]),

    # ---------------- DEVOPS ----------------
    ("Configured Docker containers for deployment", ["devops"]),
    ("Set up CI/CD pipeline using GitHub Actions", ["devops"]),
    ("Built Docker images for microservices", ["devops"]),
    ("Worked on Jenkins CI automation", ["devops"]),
    ("Configured container orchestration tasks", ["devops"]),
    ("Automated deployments using shell scripts", ["devops"]),
    ("Managed CI/CD pipeline and deployment issues", ["devops"]),
]

texts = [t for t, labels in TRAIN_DATA]
label_lists = [labels for t, labels in TRAIN_DATA]

ALL_LABELS = sorted({label for labels in label_lists for label in labels})

mlb = MultiLabelBinarizer(classes=ALL_LABELS)
Y = mlb.fit_transform(label_lists)

pipeline = Pipeline(
    steps=[
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", OneVsRestClassifier(LogisticRegression(max_iter=2000))),
    ]
)

print("Training skill model on", len(texts), "samples...")
pipeline.fit(texts, Y)
print("Training complete.")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "skills_model.pkl")
MLB_PATH = os.path.join(MODELS_DIR, "skills_mlb.pkl")

joblib.dump(pipeline, MODEL_PATH)
joblib.dump(mlb, MLB_PATH)

print("Saved model to:", MODEL_PATH)
print("Saved label encoder to:", MLB_PATH)
print("Skill labels:", ALL_LABELS)
