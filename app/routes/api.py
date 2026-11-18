from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Prediction, Student
from app.schemas import StudentCreate, StudentOut
from app.services.predict import build_prediction_report

router = APIRouter(prefix="/api", tags=["Placement API"])


def _get_artifact(request: Request) -> dict:
    artifact = getattr(request.app.state, "ml_artifact", None)
    if artifact is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model and dataset are not loaded yet. Ensure internet access on first run or place the CSV in data/.",
        )
    return artifact


@router.get("/health")
def health_check(request: Request):
    artifact = getattr(request.app.state, "ml_artifact", None)
    return {
        "status": "ok",
        "model_ready": artifact is not None,
        "dataset_loaded": artifact is not None,
        "total_dataset_records": artifact["metrics"]["total_records"] if artifact else 0,
        "message": (
            "API and Random Forest pipeline ready."
            if artifact
            else "API running, waiting for dataset CSV & model training."
        ),
    }


@router.get("/students", response_model=list[StudentOut])
def list_students(db: Session = Depends(get_db)):
    return db.query(Student).order_by(Student.created_at.desc()).all()


@router.post("/students", response_model=StudentOut, status_code=status.HTTP_201_CREATED)
def create_student(payload: StudentCreate, db: Session = Depends(get_db)):
    normalized_email = payload.email.strip().lower()
    existing = db.query(Student).filter(func.lower(Student.email) == normalized_email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"A student profile with email '{normalized_email}' already exists.",
        )

    data = payload.model_dump()
    data["email"] = normalized_email
    student = Student(**data)
    db.add(student)
    db.commit()
    db.refresh(student)
    return student


@router.delete("/students/{sid}")
def delete_student(sid: int, db: Session = Depends(get_db)):
    student = db.query(Student).filter(Student.id == sid).first()
    if not student:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found.")
    db.delete(student)
    db.commit()
    return {"message": "Student profile and linked predictions deleted.", "deleted_id": sid}


@router.post("/predict/{sid}")
def predict_student(sid: int, request: Request, db: Session = Depends(get_db)):
    artifact = _get_artifact(request)
    student = db.query(Student).filter(Student.id == sid).first()
    if not student:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found.")

    report = build_prediction_report(student, artifact)

    record = Prediction(
        student_id=student.id,
        prediction_label=report["prediction"],
        score=report["score"],
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    report["prediction_id"] = record.id
    report["created_at"] = record.created_at.isoformat()
    return report


@router.get("/history")
def prediction_history(db: Session = Depends(get_db)):
    rows = (
        db.query(Prediction, Student)
        .join(Student, Prediction.student_id == Student.id)
        .order_by(Prediction.created_at.desc())
        .limit(100)
        .all()
    )
    return [
        {
            "id": pred.id,
            "student_id": stud.id,
            "student_name": stud.full_name,
            "email": stud.email,
            "programme": stud.programme,
            "cgpa": stud.cgpa,
            "prediction": pred.prediction_label,
            "score": pred.score,
            "created_at": pred.created_at.isoformat(),
        }
        for pred, stud in rows
    ]


@router.get("/summary")
def dashboard_summary(db: Session = Depends(get_db)):
    student_count = db.query(func.count(Student.id)).scalar() or 0
    prediction_count = db.query(func.count(Prediction.id)).scalar() or 0
    avg_cgpa = db.query(func.avg(Student.cgpa)).scalar()
    placed_predictions = (
        db.query(func.count(Prediction.id)).filter(Prediction.prediction_label == "Placed").scalar() or 0
    )
    not_placed_predictions = (
        db.query(func.count(Prediction.id)).filter(Prediction.prediction_label == "Not Placed").scalar() or 0
    )

    recent_rows = (
        db.query(Prediction, Student)
        .join(Student, Prediction.student_id == Student.id)
        .order_by(Prediction.created_at.desc())
        .limit(8)
        .all()
    )
    recent_history = [
        {
            "id": pred.id,
            "student_id": stud.id,
            "student_name": stud.full_name,
            "programme": stud.programme,
            "prediction": pred.prediction_label,
            "score": pred.score,
            "created_at": pred.created_at.isoformat(),
        }
        for pred, stud in recent_rows
    ]

    return {
        "student_count": int(student_count),
        "prediction_count": int(prediction_count),
        "average_cgpa": round(float(avg_cgpa), 2) if avg_cgpa is not None else 0.0,
        "placed_predictions": int(placed_predictions),
        "not_placed_predictions": int(not_placed_predictions),
        "recent_history": recent_history,
    }


@router.get("/metrics")
def model_metrics(request: Request):
    artifact = _get_artifact(request)
    return artifact["metrics"]


@router.get("/analytics")
def analytics_overview(request: Request, db: Session = Depends(get_db)):
    artifact = _get_artifact(request)

    students = db.query(Student).all()
    branch_counts: dict[str, int] = {}
    cgpa_bands = {"< 5.0": 0, "5.0–6.0": 0, "6.0–7.0": 0, "7.0–8.0": 0, "8.0–9.0": 0, "> 9.0": 0}

    for s in students:
        branch_counts[s.programme] = branch_counts.get(s.programme, 0) + 1
        if s.cgpa < 5.0:
            cgpa_bands["< 5.0"] += 1
        elif s.cgpa < 6.0:
            cgpa_bands["5.0–6.0"] += 1
        elif s.cgpa < 7.0:
            cgpa_bands["6.0–7.0"] += 1
        elif s.cgpa < 8.0:
            cgpa_bands["7.0–8.0"] += 1
        elif s.cgpa <= 9.0:
            cgpa_bands["8.0–9.0"] += 1
        else:
            cgpa_bands["> 9.0"] += 1

    placed_preds = (
        db.query(func.count(Prediction.id)).filter(Prediction.prediction_label == "Placed").scalar() or 0
    )
    not_placed_preds = (
        db.query(func.count(Prediction.id)).filter(Prediction.prediction_label == "Not Placed").scalar() or 0
    )

    return {
        "dataset": artifact["analytics"],
        "application": {
            "total_profiles": len(students),
            "total_predictions": int(placed_preds + not_placed_preds),
            "profiles_by_branch": [
                {"programme": k, "count": v}
                for k, v in sorted(branch_counts.items(), key=lambda x: x[1], reverse=True)
            ],
            "cgpa_distribution": [{"band": k, "count": v} for k, v in cgpa_bands.items()],
            "prediction_outcomes": {
                "placed": int(placed_preds),
                "not_placed": int(not_placed_preds),
            },
        },
    }