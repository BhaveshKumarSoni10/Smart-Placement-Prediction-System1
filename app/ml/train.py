from __future__ import annotations

import logging
import re
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.config import DATASET_PATH, DATASET_URLS, MODEL_PATH, RANDOM_SEED

logger = logging.getLogger(__name__)

# Canonical feature names -> recognised source column aliases (case-insensitive)
COLUMN_ALIASES: dict[str, list[str]] = {
    "programme": ["major", "programme", "program", "branch", "department", "specialization", "stream"],
    "cgpa": ["cgpa", "gpa", "cumulative_gpa"],
    "academic_performance": ["academic_performance", "academic_score", "academics"],
    "attendance": ["attendance_percentage", "attendance", "attendance_pct", "attendance_rate"],
    "study_hours_per_week": ["study_hours_per_week", "study_hours", "weekly_study_hours"],
    "programming_skill": ["programming_skill", "programming_skills", "coding_skill", "coding_skills"],
    "projects_completed": ["projects_completed", "projects", "num_projects", "project_count"],
    "certifications": ["certifications", "certification_count", "num_certifications"],
    "hackathons": ["hackathons", "hackathon_count", "hackathons_participated"],
    "internship_experience": ["internships", "internship_experience", "internship", "has_internship"],
    "leadership_experience": ["leadership_experience", "leadership", "leadership_score"],
    "github_repositories": ["github_profile", "github_repositories", "github_repos", "github_activity"],
    "linkedin_profile": ["linkedin_profile", "linkedin", "has_linkedin"],
    "resume_score": ["resume_score", "resume_quality", "cv_score"],
    "communication_skills": ["communication_skills", "communication", "communication_score"],
    "teamwork": ["teamwork", "team_work", "teamwork_score"],
    "problem_solving": ["problem_solving", "problem_solving_skills", "aptitude"],
    "english_proficiency": ["english_proficiency", "english", "english_score"],
    "university_year": ["university_year", "year_of_study", "academic_year", "year"],
    "target": ["placement_status", "placed", "placement", "is_placed", "career_outcome", "status"],
}

CANONICAL_FEATURES = [k for k in COLUMN_ALIASES if k != "target"]

PROGRAMME_NORMALIZATION = {
    "ai & ds": "Artificial Intelligence and Data Science (AI & DS)",
    "aids": "Artificial Intelligence and Data Science (AI & DS)",
    "ai and ds": "Artificial Intelligence and Data Science (AI & DS)",
    "artificial intelligence and data science": "Artificial Intelligence and Data Science (AI & DS)",
    "ai & ml": "Artificial Intelligence and Machine Learning (AI & ML)",
    "aiml": "Artificial Intelligence and Machine Learning (AI & ML)",
    "ai and ml": "Artificial Intelligence and Machine Learning (AI & ML)",
    "artificial intelligence and machine learning": "Artificial Intelligence and Machine Learning (AI & ML)",
    "artificial intelligence": "Artificial Intelligence and Machine Learning (AI & ML)",
    "data science": "Data Science",
    "ds": "Data Science",
    "cyber security": "Cyber Security",
    "cybersecurity": "Cyber Security",
    "cs": "Computer Science",
    "cse": "Computer Science",
    "computer science": "Computer Science",
    "computer science and engineering": "Computer Science",
    "it": "Information Technology",
    "information technology": "Information Technology",
    "ece": "Electronics and Communication Engineering",
    "electronics and communication": "Electronics and Communication Engineering",
    "electronics and communication engineering": "Electronics and Communication Engineering",
    "electronics": "Electronics Engineering",
    "electronics engineering": "Electronics Engineering",
    "me": "Mechanical Engineering",
    "mechanical": "Mechanical Engineering",
    "mechanical engineering": "Mechanical Engineering",
    "ce": "Civil Engineering",
    "civil": "Civil Engineering",
    "civil engineering": "Civil Engineering",
    "ee": "Electrical Engineering",
    "electrical": "Electrical Engineering",
    "electrical engineering": "Electrical Engineering",
    "mba": "MBA",
    "master of business administration": "MBA",
    "bba": "BBA",
    "bachelor of business administration": "BBA",
    "msc": "MSc",
    "m.sc": "MSc",
    "bsc": "BSc",
    "b.sc": "BSc",
    "bcom": "BCom",
    "b.com": "BCom",
    "business": "BBA",
    "finance": "BCom",
    "commerce": "BCom",
    "engineering": "Other",
    "other": "Other",
}

HUMAN_FEATURE_LABELS = {
    "programme": "Programme / Branch",
    "cgpa": "CGPA",
    "academic_performance": "Academic Performance",
    "attendance": "Attendance %",
    "study_hours_per_week": "Study Hours / Week",
    "programming_skill": "Programming Skill",
    "projects_completed": "Projects Completed",
    "certifications": "Certifications",
    "hackathons": "Hackathons",
    "internship_experience": "Internship Experience",
    "leadership_experience": "Leadership Experience",
    "github_repositories": "GitHub Activity / Repos",
    "linkedin_profile": "LinkedIn Profile",
    "resume_score": "Resume Score",
    "communication_skills": "Communication Skills",
    "teamwork": "Teamwork",
    "problem_solving": "Problem Solving",
    "english_proficiency": "English Proficiency",
    "university_year": "University Year",
}


def download_dataset_if_needed() -> bool:
    if DATASET_PATH.exists() and DATASET_PATH.stat().st_size > 1000:
        return True

    for url in DATASET_URLS:
        try:
            logger.info("Downloading placement dataset from %s ...", url)
            resp = requests.get(url, timeout=45)
            if resp.status_code == 200 and len(resp.content) > 1000:
                DATASET_PATH.write_bytes(resp.content)
                logger.info("Dataset saved to %s", DATASET_PATH)
                return True
        except Exception as exc:
            logger.warning("Download attempt failed from %s: %s", url, exc)

    logger.error(
        "Unable to download dataset automatically. Place CSV manually at %s",
        DATASET_PATH,
    )
    return False


def normalise_programme(value: Any) -> str:
    if pd.isna(value):
        return "Other"
    raw = str(value).strip()
    if not raw:
        return "Other"
    key = re.sub(r"\s+", " ", raw.lower()).strip()
    if key in PROGRAMME_NORMALIZATION:
        return PROGRAMME_NORMALIZATION[key]
    for pattern, mapped in PROGRAMME_NORMALIZATION.items():
        if pattern == key or (len(pattern) > 3 and pattern in key):
            return mapped
    return raw


def clean_target(series: pd.Series) -> pd.Series:
    def _map(val: Any) -> str | None:
        if pd.isna(val):
            return None
        s = str(val).strip().lower()
        if s in {"placed", "1", "yes", "true", "hired", "selected"}:
            return "Placed"
        if s in {"not placed", "not_placed", "unplaced", "0", "no", "false", "rejected"}:
            return "Not Placed"
        return None

    return series.apply(_map)


def match_columns(df: pd.DataFrame) -> dict[str, str]:
    lower_to_actual = {c.strip().lower().replace(" ", "_"): c for c in df.columns}
    matched: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lower_to_actual:
                matched[canonical] = lower_to_actual[alias]
                break
    return matched


def _is_binary_like(series: pd.Series) -> bool:
    vals = series.dropna().astype(str).str.strip().str.lower().unique()
    if len(vals) == 0:
        return False
    binary_tokens = {"yes", "no", "true", "false", "0", "1", "y", "n"}
    return all(v in binary_tokens for v in vals)


def _to_yes_no(val: Any) -> str:
    if pd.isna(val):
        return "No"
    if isinstance(val, (int, float)):
        return "Yes" if float(val) > 0 else "No"
    s = str(val).strip().lower()
    if s in {"yes", "true", "1", "y"}:
        return "Yes"
    if s in {"no", "false", "0", "n", ""}:
        return "No"
    try:
        return "Yes" if float(s) > 0 else "No"
    except ValueError:
        return "No"


def prepare_dataframe(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    col_map = match_columns(raw_df)
    if "target" not in col_map:
        raise ValueError("Could not find a recognised placement target column in the dataset.")

    df = pd.DataFrame()
    df["target"] = clean_target(raw_df[col_map["target"]])
    df = df.dropna(subset=["target"]).copy()
    valid_idx = df.index

    feature_meta: dict[str, Any] = {}
    categorical_cols: list[str] = []
    numeric_cols: list[str] = []

    for feat in CANONICAL_FEATURES:
        if feat not in col_map:
            continue
        src = raw_df.loc[valid_idx, col_map[feat]]

        if feat == "programme":
            df[feat] = src.apply(normalise_programme).astype(str)
            categorical_cols.append(feat)
            feature_meta[feat] = {"kind": "categorical"}
        elif feat == "university_year":
            df[feat] = src.fillna("Unknown").astype(str).str.strip()
            categorical_cols.append(feat)
            feature_meta[feat] = {"kind": "categorical"}
        elif feat in {"internship_experience", "linkedin_profile", "github_repositories", "leadership_experience"}:
            # Some CSV versions encode these as Yes/No strings, others as numeric counts/scores
            num_attempt = pd.to_numeric(src, errors="coerce")
            if _is_binary_like(src) or num_attempt.notna().mean() < 0.5:
                df[feat] = src.apply(_to_yes_no)
                categorical_cols.append(feat)
                feature_meta[feat] = {"kind": "yes_no"}
            else:
                df[feat] = num_attempt
                numeric_cols.append(feat)
                max_val = float(num_attempt.dropna().max()) if num_attempt.notna().any() else 10.0
                scale = 100.0 if (feat == "leadership_experience" and max_val > 12.0) else 10.0
                feature_meta[feat] = {"kind": "numeric", "dataset_scale": scale, "max_observed": max_val}
        else:
            num_series = pd.to_numeric(src, errors="coerce")
            df[feat] = num_series
            numeric_cols.append(feat)
            max_val = float(num_series.dropna().max()) if num_series.notna().any() else 10.0
            scale = 10.0
            if feat in {"academic_performance", "programming_skill", "resume_score", "communication_skills", "teamwork", "problem_solving", "english_proficiency"}:
                scale = 100.0 if max_val > 12.0 else 10.0
            elif feat == "attendance":
                scale = 100.0
            elif feat == "cgpa":
                scale = 10.0 if max_val <= 10.5 else 100.0
            feature_meta[feat] = {"kind": "numeric", "dataset_scale": scale, "max_observed": max_val}

    meta = {
        "matched_columns": col_map,
        "feature_meta": feature_meta,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
    }
    return df, meta


def _band_rate(df: pd.DataFrame, col: str, bins: list[float], labels: list[str]) -> list[dict[str, Any]]:
    if col not in df.columns:
        return []
    series = pd.to_numeric(df[col], errors="coerce")
    if series.dropna().empty:
        return []
    # Scale 0-100 academic_performance / programming_skill down to 0-10 bands if needed
    if col in {"academic_performance", "programming_skill"} and series.max() > 12:
        series = series / 10.0
    if col == "cgpa" and series.max() > 12:
        series = series / 10.0

    binned = pd.cut(series, bins=bins, labels=labels, include_lowest=True, right=True)
    is_placed = (df["target"] == "Placed").astype(int)
    grouped = pd.DataFrame({"band": binned, "placed": is_placed}).dropna(subset=["band"])
    summary = grouped.groupby("band", observed=False)["placed"].agg(["mean", "count"]).reset_index()
    return [
        {
            "band": str(row["band"]),
            "rate": round(float(row["mean"]) * 100.0, 2) if row["count"] > 0 else 0.0,
            "count": int(row["count"]),
        }
        for _, row in summary.iterrows()
    ]


def compute_dataset_analytics(df: pd.DataFrame, feature_meta: dict[str, Any]) -> dict[str, Any]:
    is_placed = (df["target"] == "Placed").astype(int)
    placed_count = int(is_placed.sum())
    not_placed_count = int(len(df) - placed_count)

    # Programme stats
    programme_counts: dict[str, int] = {}
    by_programme_rate: list[dict[str, Any]] = []
    if "programme" in df.columns:
        prog_df = pd.DataFrame({"programme": df["programme"], "placed": is_placed})
        grouped = (
            prog_df.groupby("programme")["placed"]
            .agg(["mean", "count", "sum"])
            .sort_values(by="count", ascending=False)
        )
        for prog, row in grouped.iterrows():
            programme_counts[str(prog)] = int(row["count"])
            by_programme_rate.append(
                {
                    "programme": str(prog),
                    "rate": round(float(row["mean"]) * 100.0, 2),
                    "count": int(row["count"]),
                    "placed_count": int(row["sum"]),
                }
            )

    # Internship experience rate
    by_internship: list[dict[str, Any]] = []
    if "internship_experience" in df.columns:
        intern_label = df["internship_experience"].apply(_to_yes_no)
        grp = pd.DataFrame({"label": intern_label, "placed": is_placed}).groupby("label")["placed"].agg(["mean", "count"])
        for lbl in ["Yes", "No"]:
            if lbl in grp.index:
                by_internship.append(
                    {
                        "label": "With Internship" if lbl == "Yes" else "No Internship",
                        "rate": round(float(grp.loc[lbl, "mean"]) * 100.0, 2),
                        "count": int(grp.loc[lbl, "count"]),
                    }
                )

    # Projects completed (0..9, 10+)
    by_projects: list[dict[str, Any]] = []
    if "projects_completed" in df.columns:
        proj = pd.to_numeric(df["projects_completed"], errors="coerce").dropna().astype(int)
        proj_group = proj.apply(lambda x: "10+" if x >= 10 else str(max(0, x)))
        tmp = pd.DataFrame({"grp": proj_group, "placed": is_placed.loc[proj.index]})
        grouped = tmp.groupby("grp")["placed"].agg(["mean", "count"])
        order = [str(i) for i in range(10)] + ["10+"]
        for key in order:
            if key in grouped.index:
                by_projects.append(
                    {
                        "label": f"{key} Projects" if key == "10+" else key,
                        "rate": round(float(grouped.loc[key, "mean"]) * 100.0, 2),
                        "count": int(grouped.loc[key, "count"]),
                    }
                )

    # Numeric feature medians & 75th percentiles (converted to form display scales)
    numeric_benchmarks: dict[str, dict[str, float]] = {}
    for col, info in feature_meta.items():
        if info.get("kind") != "numeric" or col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        med = float(series.median())
        p75 = float(series.quantile(0.75))
        scale = float(info.get("dataset_scale", 10.0))
        # Convert 0-100 dataset columns (except attendance) into 0-10 form scale for consistent display
        if col in {
            "academic_performance",
            "programming_skill",
            "resume_score",
            "communication_skills",
            "teamwork",
            "problem_solving",
            "english_proficiency",
            "leadership_experience",
            "cgpa",
        } and scale == 100.0:
            med /= 10.0
            p75 /= 10.0

        numeric_benchmarks[col] = {
            "median": round(med, 2),
            "p75": round(p75, 2),
            "max": round(float(series.max()), 2),
        }

    return {
        "total_records": int(len(df)),
        "placement_distribution": {"placed": placed_count, "not_placed": not_placed_count},
        "programme_counts": programme_counts,
        "by_programme_rate": by_programme_rate,
        "by_cgpa_band": _band_rate(df, "cgpa", [0, 5, 6, 7, 8, 9, 10.5], ["< 5.0", "5.0–6.0", "6.0–7.0", "7.0–8.0", "8.0–9.0", "> 9.0"]),
        "by_internship": by_internship,
        "by_projects": by_projects,
        "by_academic_band": _band_rate(df, "academic_performance", [0, 5, 6.5, 8, 10.5], ["< 5.0", "5.0–6.5", "6.5–8.0", "8.0–10.0"]),
        "by_attendance_band": _band_rate(df, "attendance", [0, 59.99, 69.99, 79.99, 89.99, 100.5], ["< 60%", "60–69%", "70–79%", "80–89%", "90–100%"]),
        "by_programming_band": _band_rate(df, "programming_skill", [0, 4, 6, 8, 10.5], ["0–4", "4–6", "6–8", "8–10"]),
        "programme_representation": [
            {"programme": k, "count": v} for k, v in sorted(programme_counts.items(), key=lambda x: x[1], reverse=True)
        ],
        "numeric_benchmarks": numeric_benchmarks,
    }


def _aggregate_feature_importance(pipeline: Pipeline, numeric_cols: list[str], categorical_cols: list[str]) -> list[dict[str, Any]]:
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    clf: RandomForestClassifier = pipeline.named_steps["classifier"]
    importances = clf.feature_importances_

    aggregated: dict[str, float] = {}
    idx = 0

    for col in numeric_cols:
        if idx < len(importances):
            aggregated[col] = aggregated.get(col, 0.0) + float(importances[idx])
            idx += 1

    if categorical_cols:
        cat_transformer = preprocessor.named_transformers_["cat"]
        ohe: OneHotEncoder = cat_transformer.named_steps["encoder"]
        for col, cats in zip(categorical_cols, ohe.categories_):
            col_imp = 0.0
            for _ in cats:
                if idx < len(importances):
                    col_imp += float(importances[idx])
                    idx += 1
            aggregated[col] = aggregated.get(col, 0.0) + col_imp

    total = sum(aggregated.values()) or 1.0
    result = [
        {
            "feature_key": k,
            "feature": HUMAN_FEATURE_LABELS.get(k, k.replace("_", " ").title()),
            "importance": round((v / total) * 100.0, 2),
        }
        for k, v in aggregated.items()
    ]
    result.sort(key=lambda x: x["importance"], reverse=True)
    return result


def train_and_save_model(force_retrain: bool = False) -> dict[str, Any] | None:
    if MODEL_PATH.exists() and not force_retrain:
        try:
            artifact = joblib.load(MODEL_PATH)
            if isinstance(artifact, dict) and "pipeline" in artifact and "analytics" in artifact:
                logger.info("Loaded existing trained model artifact from %s", MODEL_PATH)
                return artifact
        except Exception as exc:
            logger.warning("Existing model unreadable (%s); retraining...", exc)

    if not download_dataset_if_needed():
        return None

    raw_df = pd.read_csv(DATASET_PATH)
    df, meta = prepare_dataframe(raw_df)

    feature_cols = meta["numeric_cols"] + meta["categorical_cols"]
    X = df[feature_cols]
    y = (df["target"] == "Placed").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, meta["numeric_cols"]),
            ("cat", categorical_transformer, meta["categorical_cols"]),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=260,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])
    logger.info("Training Random Forest pipeline on %d rows...", len(X_train))
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "total_records": int(len(df)),
        "class_balance": {
            "placed": int(y.sum()),
            "not_placed": int(len(y) - y.sum()),
        },
        "model_params": {
            "algorithm": "RandomForestClassifier",
            "n_estimators": 260,
            "min_samples_leaf": 3,
            "class_weight": "balanced_subsample",
            "split": "80% Train / 20% Stratified Test",
        },
    }

    analytics = compute_dataset_analytics(df, meta["feature_meta"])
    feature_importance = _aggregate_feature_importance(pipeline, meta["numeric_cols"], meta["categorical_cols"])

    artifact = {
        "pipeline": pipeline,
        "feature_cols": feature_cols,
        "numeric_cols": meta["numeric_cols"],
        "categorical_cols": meta["categorical_cols"],
        "feature_meta": meta["feature_meta"],
        "metrics": metrics,
        "analytics": analytics,
        "feature_importance": feature_importance,
    }

    joblib.dump(artifact, MODEL_PATH)
    logger.info("Saved model artifact to %s (ROC-AUC=%.4f)", MODEL_PATH, metrics["roc_auc"])
    return artifact