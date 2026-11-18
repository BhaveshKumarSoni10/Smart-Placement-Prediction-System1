from __future__ import annotations

from typing import Any
import pandas as pd

from app.config import LOW_COVERAGE_THRESHOLD
from app.models import Student

BENCHMARK_SPECS = [
    ("cgpa", "CGPA", "/ 10", 10.0),
    ("academic_performance", "Academic Performance", "/ 10", 10.0),
    ("attendance", "Attendance", "%", 100.0),
    ("study_hours_per_week", "Study Hours / Week", "hrs", 60.0),
    ("programming_skill", "Programming Skill", "/ 10", 10.0),
    ("projects_completed", "Projects Completed", "count", 12.0),
    ("certifications", "Certifications", "count", 10.0),
    ("hackathons", "Hackathons", "count", 8.0),
    ("resume_score", "Resume Score", "/ 10", 10.0),
    ("communication_skills", "Communication", "/ 10", 10.0),
    ("problem_solving", "Problem Solving", "/ 10", 10.0),
    ("english_proficiency", "English Proficiency", "/ 10", 10.0),
]


def student_to_model_row(student: Student, feature_cols: list[str], feature_meta: dict[str, Any]) -> dict[str, Any]:
    raw_map: dict[str, Any] = {
        "programme": student.programme,
        "cgpa": student.cgpa,
        "academic_performance": student.academic_performance,
        "attendance": student.attendance,
        "study_hours_per_week": student.study_hours_per_week,
        "programming_skill": student.programming_skill,
        "projects_completed": student.projects_completed,
        "certifications": student.certifications,
        "hackathons": student.hackathons,
        "internship_experience": student.internship_experience,
        "leadership_experience": student.leadership_experience,
        "github_repositories": student.github_repositories,
        "linkedin_profile": student.linkedin_profile,
        "resume_score": student.resume_score,
        "communication_skills": student.communication_skills,
        "teamwork": student.teamwork,
        "problem_solving": student.problem_solving,
        "english_proficiency": student.english_proficiency,
        "university_year": student.university_year,
    }

    row: dict[str, Any] = {}
    for col in feature_cols:
        info = feature_meta.get(col, {})
        kind = info.get("kind", "numeric")
        val = raw_map.get(col)

        if kind == "yes_no":
            if col == "internship_experience":
                row[col] = "Yes" if str(val).lower() == "yes" else "No"
            elif col == "linkedin_profile":
                row[col] = "Yes" if str(val).lower() == "yes" else "No"
            elif col == "github_repositories":
                row[col] = "Yes" if (val or 0) > 0 else "No"
            elif col == "leadership_experience":
                row[col] = "Yes" if float(val or 0) >= 5.0 else "No"
            else:
                row[col] = "Yes" if val else "No"
        elif kind == "numeric":
            if col == "internship_experience":
                row[col] = 1.0 if str(val).lower() == "yes" else 0.0
            else:
                num = float(val or 0.0)
                if info.get("dataset_scale") == 100.0 and col != "attendance":
                    num *= 10.0
                row[col] = num
        else:
            row[col] = str(val) if val is not None else "Other"

    return row


def _score_row(pipeline: Any, row: dict[str, Any], feature_cols: list[str]) -> tuple[str, float]:
    df_row = pd.DataFrame([row], columns=feature_cols)
    prob_placed = float(pipeline.predict_proba(df_row)[0][1])
    pred_idx = int(pipeline.predict(df_row)[0])
    label = "Placed" if pred_idx == 1 else "Not Placed"
    return label, round(prob_placed * 100.0, 2)


def generate_recommendations(student: Student) -> list[dict[str, str]]:
    recs: list[dict[str, str]] = []

    if student.backlogs > 0:
        recs.append(
            {
                "priority": "High",
                "area": "Pending Backlogs",
                "current_value": f"{student.backlogs} backlog(s)",
                "message": "Pending backlogs frequently restrict eligibility for campus placement drives.",
                "action_step": "Prioritise clearing pending papers in the upcoming exam cycle and map out subject-wise revision blocks.",
            }
        )

    if student.cgpa < 7.0 or student.academic_performance < 7.0:
        recs.append(
            {
                "priority": "High",
                "area": "CGPA & Academic Consistency",
                "current_value": f"CGPA {student.cgpa:.2f} | Academic {student.academic_performance:.1f}/10",
                "message": "Academic indicators are below typical shortlisting comfort zones.",
                "action_step": "Identify lowest-scoring core subjects, solve past exam papers weekly, and seek faculty feedback on weak topics.",
            }
        )

    if student.internship_experience == "No":
        recs.append(
            {
                "priority": "High",
                "area": "Practical / Internship Exposure",
                "current_value": "No Internship",
                "message": "Hands-on industry or supervised research exposure strengthens employability profiles significantly.",
                "action_step": "Apply for short-term internships, open-source contributions, or faculty-guided applied projects this semester.",
            }
        )

    if student.projects_completed < 3:
        recs.append(
            {
                "priority": "High",
                "area": "Project Portfolio",
                "current_value": f"{student.projects_completed} project(s)",
                "message": "Having fewer than 3 substantial projects limits evidence of applied problem-solving.",
                "action_step": "Build 2–3 end-to-end projects with live demos, clean GitHub READMEs, and clear descriptions of your personal role.",
            }
        )

    if student.programming_skill < 6.5 or student.problem_solving < 6.5:
        recs.append(
            {
                "priority": "Medium" if student.programme in {"MBA", "BBA", "BCom"} else "High",
                "area": "Programming & Problem Solving",
                "current_value": f"Prog {student.programming_skill:.1f}/10 | PS {student.problem_solving:.1f}/10",
                "message": "Technical assessments heavily test structured thinking, debugging, and core coding fluency.",
                "action_step": "Dedicate 45–60 minutes daily to role-relevant coding, data structures, or analytical case problems.",
            }
        )

    if student.resume_score < 7.0:
        recs.append(
            {
                "priority": "Medium",
                "area": "Resume Quality",
                "current_value": f"{student.resume_score:.1f}/10",
                "message": "An unstructured resume can hide strong coursework and project achievements.",
                "action_step": "Refactor into a 1-page ATS-friendly format using measurable impact verbs and quantified project outcomes.",
            }
        )

    if student.communication_skills < 7.0 or student.english_proficiency < 7.0:
        recs.append(
            {
                "priority": "Medium",
                "area": "Communication & Interview Readiness",
                "current_value": f"Comm {student.communication_skills:.1f}/10 | Eng {student.english_proficiency:.1f}/10",
                "message": "Clear explanation of projects and structured interview answers often differentiate finalists.",
                "action_step": "Practise explaining your top 2 projects aloud in 2-minute STAR format and schedule weekly mock interviews.",
            }
        )

    if student.attendance < 75.0 or student.study_hours_per_week < 10.0:
        recs.append(
            {
                "priority": "Medium",
                "area": "Attendance & Study Rhythm",
                "current_value": f"{student.attendance:.0f}% att. | {student.study_hours_per_week:.0f} hrs/wk",
                "message": "Irregular attendance or low weekly study hours may affect continuous assessment and placement eligibility.",
                "action_step": "Maintain ≥80% attendance and block consistent weekly hours for core coursework and placement prep.",
            }
        )

    if student.certifications < 1 or student.hackathons < 1 or student.github_repositories < 2 or student.linkedin_profile == "No":
        recs.append(
            {
                "priority": "Low",
                "area": "Certifications, Hackathons & Digital Presence",
                "current_value": f"{student.certifications} certs | {student.hackathons} hackathons | {student.github_repositories} repos",
                "message": "Verified credentials and active public portfolios reinforce credibility with recruiters.",
                "action_step": "Complete 1 role-aligned certification, participate in a hackathon/case challenge, and keep LinkedIn/GitHub updated.",
            }
        )

    if not recs:
        recs.append(
            {
                "priority": "Low",
                "area": "Maintenance & Mock Interviews",
                "current_value": "Well-balanced profile",
                "message": "Your profile indicators align well with general placement readiness benchmarks.",
                "action_step": "Focus on company-specific mock interviews, system/domain depth, and keeping projects well-documented.",
            }
        )

    order = {"High": 0, "Medium": 1, "Low": 2}
    recs.sort(key=lambda r: order.get(r["priority"], 3))
    return recs


def build_prediction_report(student: Student, artifact: dict[str, Any]) -> dict[str, Any]:
    pipeline = artifact["pipeline"]
    feature_cols: list[str] = artifact["feature_cols"]
    feature_meta: dict[str, Any] = artifact["feature_meta"]
    analytics: dict[str, Any] = artifact["analytics"]
    numeric_benchmarks: dict[str, dict[str, float]] = analytics.get("numeric_benchmarks", {})
    programme_counts: dict[str, int] = analytics.get("programme_counts", {})

    base_row = student_to_model_row(student, feature_cols, feature_meta)
    label, base_score = _score_row(pipeline, base_row, feature_cols)

    # Programme coverage
    prog_count = int(programme_counts.get(student.programme, 0))
    is_low_coverage = prog_count < LOW_COVERAGE_THRESHOLD
    if prog_count == 0:
        coverage_note = (
            f"Zero training records matched '{student.programme}'. The model falls back to "
            "other features and out-of-category defaults; interpret this score with high caution."
        )
    elif is_low_coverage:
        coverage_note = (
            f"Only {prog_count} training records exist for '{student.programme}' (below the "
            f"{LOW_COVERAGE_THRESHOLD}-record guidance threshold). Model evidence is limited."
        )
    else:
        coverage_note = (
            f"{prog_count:,} synthetic training records exist for '{student.programme}'. "
            "Remember this reflects synthetic training coverage, not verified campus hiring data."
        )

    # Benchmarks & paired chart data
    benchmarks: list[dict[str, Any]] = []
    for key, title, unit, visual_max in BENCHMARK_SPECS:
        student_val = float(getattr(student, key))
        bm = numeric_benchmarks.get(key)
        if not bm:
            continue
        med = float(bm["median"])
        p75 = float(bm["p75"])
        scale_max = max(visual_max, student_val, p75, 1.0)
        benchmarks.append(
            {
                "key": key,
                "label": title,
                "unit": unit,
                "student_value": round(student_val, 2),
                "dataset_median": round(med, 2),
                "dataset_p75": round(p75, 2),
                "student_norm": round(min(100.0, (student_val / scale_max) * 100.0), 2),
                "median_norm": round(min(100.0, (med / scale_max) * 100.0), 2),
            }
        )

    # One-feature-at-a-time What-If improvement scenarios
    what_if_candidates = [
        ("cgpa", "CGPA → 8.5", 8.5),
        ("internship_experience", "Internship → Yes", "Yes"),
        ("programming_skill", "Programming Skill → 8.5", 8.5),
        ("projects_completed", "Projects → 5", 5),
        ("resume_score", "Resume Score → 8.5", 8.5),
        ("communication_skills", "Communication → 8.5", 8.5),
        ("problem_solving", "Problem Solving → 8.5", 8.5),
        ("attendance", "Attendance → 90%", 90.0),
        ("certifications", "Certifications → 3", 3),
    ]

    what_if: list[dict[str, Any]] = []
    for col, desc, target_val in what_if_candidates:
        if col not in feature_cols:
            continue
        mod_row = dict(base_row)
        info = feature_meta.get(col, {})
        kind = info.get("kind")

        if kind == "yes_no":
            if mod_row[col] == "Yes":
                continue
            mod_row[col] = "Yes"
        elif kind == "numeric":
            if col == "internship_experience":
                if float(mod_row[col]) >= 1.0:
                    continue
                mod_row[col] = 1.0
            else:
                scaled_target = float(target_val)
                if info.get("dataset_scale") == 100.0 and col != "attendance":
                    scaled_target *= 10.0
                if float(mod_row[col]) >= scaled_target:
                    continue
                mod_row[col] = scaled_target

        _, new_score = _score_row(pipeline, mod_row, feature_cols)
        what_if.append(
            {
                "feature_key": col,
                "label": desc,
                "current_score": base_score,
                "what_if_score": new_score,
                "delta": round(new_score - base_score, 2),
            }
        )

    what_if.sort(key=lambda x: x["delta"], reverse=True)

    # Programme representation for this report (include selected programme even if 0)
    prog_chart = [
        {"programme": p, "count": int(c), "is_selected": (p == student.programme)}
        for p, c in sorted(programme_counts.items(), key=lambda x: x[1], reverse=True)
    ]
    if student.programme not in programme_counts:
        prog_chart.append({"programme": student.programme, "count": 0, "is_selected": True})

    return {
        "student_id": student.id,
        "student_name": student.full_name,
        "programme": student.programme,
        "prediction": label,
        "score": base_score,
        "programme_coverage": {
            "programme": student.programme,
            "record_count": prog_count,
            "threshold": LOW_COVERAGE_THRESHOLD,
            "is_low_coverage": is_low_coverage,
            "coverage_note": coverage_note,
        },
        "benchmarks": benchmarks,
        "what_if": what_if[:8],
        "feature_importance": artifact.get("feature_importance", [])[:12],
        "programme_counts": prog_chart,
        "recommendations": generate_recommendations(student),
    }