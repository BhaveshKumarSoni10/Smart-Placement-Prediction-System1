from datetime import datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator
from app.config import SUPPORTED_PROGRAMMES


class StudentCreate(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=120)
    email: EmailStr
    programme: str = Field(..., min_length=2, max_length=120)

    cgpa: float = Field(..., ge=0.0, le=10.0)
    academic_performance: float = Field(..., ge=0.0, le=10.0)
    attendance: float = Field(..., ge=0.0, le=100.0)
    study_hours_per_week: float = Field(..., ge=0.0, le=112.0)

    programming_skill: float = Field(..., ge=0.0, le=10.0)
    internship_experience: Literal["Yes", "No"]
    projects_completed: int = Field(..., ge=0, le=100)
    certifications: int = Field(..., ge=0, le=100)
    hackathons: int = Field(..., ge=0, le=100)
    leadership_experience: float = Field(..., ge=0.0, le=10.0)
    github_repositories: int = Field(..., ge=0, le=500)
    linkedin_profile: Literal["Yes", "No"]
    resume_score: float = Field(..., ge=0.0, le=10.0)

    communication_skills: float = Field(..., ge=0.0, le=10.0)
    teamwork: float = Field(..., ge=0.0, le=10.0)
    problem_solving: float = Field(..., ge=0.0, le=10.0)
    english_proficiency: float = Field(..., ge=0.0, le=10.0)

    university_year: str = Field(..., min_length=2, max_length=40)
    backlogs: int = Field(default=0, ge=0, le=50)

    @field_validator("full_name", "university_year")
    @classmethod
    def strip_text(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Field cannot be blank.")
        return cleaned

    @field_validator("programme")
    @classmethod
    def validate_programme(cls, v: str) -> str:
        cleaned = v.strip()
        if cleaned not in SUPPORTED_PROGRAMMES:
            raise ValueError("Selected programme is not in the supported list.")
        return cleaned


class StudentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    full_name: str
    email: str
    programme: str
    cgpa: float
    academic_performance: float
    attendance: float
    study_hours_per_week: float
    programming_skill: float
    internship_experience: str
    projects_completed: int
    certifications: int
    hackathons: int
    leadership_experience: float
    github_repositories: int
    linkedin_profile: str
    resume_score: float
    communication_skills: float
    teamwork: float
    problem_solving: float
    english_proficiency: float
    university_year: str
    backlogs: int
    created_at: datetime