from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from app.database import Base


class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(120), nullable=False)
    email = Column(String(180), unique=True, index=True, nullable=False)
    programme = Column(String(120), nullable=False)

    cgpa = Column(Float, nullable=False)
    academic_performance = Column(Float, nullable=False)
    attendance = Column(Float, nullable=False)
    study_hours_per_week = Column(Float, nullable=False)

    programming_skill = Column(Float, nullable=False)
    internship_experience = Column(String(10), nullable=False)  # "Yes" or "No"
    projects_completed = Column(Integer, nullable=False)
    certifications = Column(Integer, nullable=False)
    hackathons = Column(Integer, nullable=False)
    leadership_experience = Column(Float, nullable=False)
    github_repositories = Column(Integer, nullable=False)
    linkedin_profile = Column(String(10), nullable=False)  # "Yes" or "No"
    resume_score = Column(Float, nullable=False)

    communication_skills = Column(Float, nullable=False)
    teamwork = Column(Float, nullable=False)
    problem_solving = Column(Float, nullable=False)
    english_proficiency = Column(Float, nullable=False)

    university_year = Column(String(40), nullable=False)
    backlogs = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    predictions = relationship(
        "Prediction",
        back_populates="student",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id", ondelete="CASCADE"), nullable=False, index=True)
    prediction_label = Column(String(20), nullable=False)  # "Placed" or "Not Placed"
    score = Column(Float, nullable=False)  # 0 - 100 experimental display score
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    student = relationship("Student", back_populates="predictions")