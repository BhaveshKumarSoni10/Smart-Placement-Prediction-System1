
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
FRONTEND_DIR = BASE_DIR / "frontend"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = DATA_DIR / "student_career_success_dataset.csv"
MODEL_PATH = MODELS_DIR / "placement_model.joblib"

DATASET_URLS = [
    "https://raw.githubusercontent.com/MobeenFatimaa/student-career-success-placement-prediction/main/student_career_success_dataset.csv",
    "https://raw.githubusercontent.com/MobeenFatimaa/student-career-success-placement-prediction/master/student_career_success_dataset.csv",
]

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'placement.db'}")

LOW_COVERAGE_THRESHOLD = 30
RANDOM_SEED = 42

SUPPORTED_PROGRAMMES = [
    "Artificial Intelligence and Data Science (AI & DS)",
    "Artificial Intelligence and Machine Learning (AI & ML)",
    "Data Science",
    "Cyber Security",
    "Computer Science",
    "Information Technology",
    "Electronics Engineering",
    "Electronics and Communication Engineering",
    "Mechanical Engineering",
    "Civil Engineering",
    "Electrical Engineering",
    "MBA",
    "BBA",
    "MSc",
    "BSc",
    "BCom",
    "Other",
]