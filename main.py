# ================================================================
# main.py — FINAL VERSION
# FastAPI backend for XAI Resume Screening
# Run with: python -m uvicorn main:app --reload
# ================================================================

import json
import numpy as np
import pandas as pd
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from parser import parse_resume


# ----------------------------------------------------------------
# LOAD ARTIFACTS
# ----------------------------------------------------------------
try:
    model    = joblib.load('model.pkl')
    scaler   = joblib.load('scaler.pkl')
    FEATURES = joblib.load('features.pkl')

    with open('config.json', 'r') as f:
        config = json.load(f)

    THRESHOLD      = config['threshold']
    NUMERICAL_COLS = config['numerical_cols']
    ENCODING_MAPS  = config['encoding_maps']
    HARD_RULES     = config.get('hard_rules', {})

    explainer = shap.TreeExplainer(model)
    print(f"Model loaded | Threshold: {THRESHOLD:.4f} | Features: {len(FEATURES)}")

except Exception as e:
    raise RuntimeError(f"Failed to load model artifacts: {e}")


# ----------------------------------------------------------------
# HARD RULES CHECKER
# ----------------------------------------------------------------
def check_hard_rules(features: dict) -> tuple:
    cgpa   = features.get('cgpa', 0)
    skills = features.get('skills_score', 0)
    prog   = features.get('programming_languages', 0)
    soft   = features.get('soft_skills_score', 0)
    intern = features.get('internships', 0)
    exp    = features.get('experience_years', 0)

    if cgpa < HARD_RULES.get('cgpa_min', 6.5):
        return False, (f"CGPA {cgpa:.2f} is below the minimum "
                       f"threshold of {HARD_RULES.get('cgpa_min', 6.5)}")

    if skills < HARD_RULES.get('skills_score_min', 6    ):
        return False, (f"Skills score {skills:.1f} is below the minimum "
                       f"threshold of {HARD_RULES.get('skills_score_min', 6)}")

    if prog < HARD_RULES.get('programming_langs_min', 2):
        return False, (f"Only {prog} programming language(s) detected — "
                       f"minimum is {HARD_RULES.get('programming_langs_min', 2)}")

    if soft < HARD_RULES.get('soft_skills_min', 4.0):
        return False, (f"Soft skills score {soft:.1f} is below the minimum "
                       f"threshold of {HARD_RULES.get('soft_skills_min', 4.0)}")

    min_exp = HARD_RULES.get('min_experience_if_no_intern', 0.5)
    if intern == 0 and exp < min_exp:
        return False, (f"No internships and only {exp:.1f} years experience "
                       f"— need at least one form of practical exposure")

    return True, "Passed all screening criteria"


# ----------------------------------------------------------------
# DATASET STATS FOR NORMALIZATION
# Must match training data ranges exactly
# ----------------------------------------------------------------
DATASET_STATS = {
    'skills_score'         : {'min': 2.0,  'max': 38.5},
    'cgpa'                 : {'min': 4.15, 'max': 11.23},
    'experience_years'     : {'min': 0.0,  'max': 23.55},
    'internships'          : {'min': 0,    'max': 10},
    'projects'             : {'min': 0,    'max': 20},
    'edu_enc'              : {'min': 1,    'max': 3},
    'tier_enc'             : {'min': 1,    'max': 3},
    'programming_languages': {'min': 1,    'max': 5},
    'hackathons'           : {'min': 0,    'max': 10},
    'research_papers'      : {'min': 0,    'max': 10},
    'certifications'       : {'min': 0,    'max': 10},
}


def norm_val(val, feature):
    """
    Normalize a single value using training dataset stats.
    Clamps to dataset min/max first to prevent out-of-range values.
    """
    stats = DATASET_STATS.get(feature, {'min': 0, 'max': 1})
    val   = max(stats['min'], min(stats['max'], float(val)))
    denom = stats['max'] - stats['min'] + 1e-9
    return (val - stats['min']) / denom


# ----------------------------------------------------------------
# FEATURE ENGINEERING (mirrors Colab Cell 3+4 exactly)
# ----------------------------------------------------------------
def engineer_features(raw: dict) -> dict:
    n_skills   = norm_val(raw['skills_score'],          'skills_score')
    n_cgpa     = norm_val(raw['cgpa'],                  'cgpa')
    n_exp      = norm_val(raw['experience_years'],      'experience_years')
    n_intern   = norm_val(raw['internships'],           'internships')
    n_proj     = norm_val(raw['projects'],              'projects')
    n_prog     = norm_val(raw['programming_languages'], 'programming_languages')
    n_hack     = norm_val(raw['hackathons'],            'hackathons')
    n_research = norm_val(raw['research_papers'],       'research_papers')
    n_cert     = norm_val(raw['certifications'],        'certifications')
    n_edu      = norm_val(raw['edu_enc'],               'edu_enc')
    n_tier     = norm_val(raw['tier_enc'],              'tier_enc')

    return {
        **raw,
        'academic_strength'  : n_cgpa * n_tier,
        'practical_exposure' : n_skills*0.5 + n_intern*0.3 + n_proj*0.2,
        'career_momentum'    : n_exp * (1 + n_intern),
        'achievement_score'  : n_hack*0.4 + n_research*0.4 + n_cert*0.2,
        'technical_breadth'  : n_skills*0.4 + n_prog*0.6,
    }


def build_feature_row(engineered: dict) -> pd.DataFrame:
    row       = {feat: engineered.get(feat, 0.0) for feat in FEATURES}
    df        = pd.DataFrame([row])
    df_scaled = df.copy()
    cols_to_scale = [c for c in NUMERICAL_COLS if c in df_scaled.columns]
    df_scaled[cols_to_scale] = scaler.transform(df_scaled[cols_to_scale])
    return df_scaled


# ----------------------------------------------------------------
# SHAP EXPLANATION
# ----------------------------------------------------------------
FEATURE_LABELS = {
    'skills_score'         : 'Technical Skills Score',
    'cgpa'                 : 'CGPA',
    'experience_years'     : 'Experience (years)',
    'internships'          : 'Internships',
    'projects'             : 'Projects',
    'programming_languages': 'Programming Languages',
    'certifications'       : 'Certifications',
    'soft_skills_score'    : 'Soft Skills Score',
    'hackathons'           : 'Hackathons',
    'research_papers'      : 'Research Papers',
    'age'                  : 'Age',
    'resume_length_words'  : 'Resume Length (words)',
    'edu_enc'              : 'Education Level',
    'tier_enc'             : 'University Tier',
    'comp_enc'             : 'Company Type',
    'academic_strength'    : 'Academic Strength (CGPA x Tier)',
    'practical_exposure'   : 'Practical Exposure',
    'career_momentum'      : 'Career Momentum',
    'achievement_score'    : 'Achievements (Hackathons + Research + Certs)',
    'technical_breadth'    : 'Technical Breadth',
}


def get_shap_explanation(df_row: pd.DataFrame, top_n: int = 4) -> dict:
    sv      = explainer.shap_values(df_row)[0]
    shap_df = pd.DataFrame({
        'feature'   : FEATURES,
        'shap_value': sv,
    }).sort_values('shap_value', ascending=False)

    positives = shap_df[shap_df['shap_value'] > 0].head(top_n)
    negatives = shap_df[shap_df['shap_value'] < 0].tail(top_n).sort_values(
        'shap_value')

    return {
        'positive_drivers': [
            {
                'feature': FEATURE_LABELS.get(r['feature'], r['feature']),
                'impact' : round(float(r['shap_value']), 4),
            }
            for _, r in positives.iterrows()
        ],
        'negative_drivers': [
            {
                'feature': FEATURE_LABELS.get(r['feature'], r['feature']),
                'impact' : round(float(r['shap_value']), 4),
            }
            for _, r in negatives.iterrows()
        ],
    }


# ----------------------------------------------------------------
# FASTAPI APP
# ----------------------------------------------------------------
app = FastAPI(
    title       = "XAI Resume Screening API",
    description = "Explainable AI resume screener — rule-based NLP + XGBoost + SHAP",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


@app.get("/")
def root():
    return {
        "message"  : "XAI Resume Screening API is running",
        "endpoints": {
            "POST /predict": "Upload a .pdf or .docx resume",
            "GET  /health" : "Health check",
            "GET  /rules"  : "View hard screening rules",
        }
    }


@app.get("/health")
def health():
    return {
        "status"   : "ok",
        "threshold": THRESHOLD,
        "features" : len(FEATURES),
    }


@app.get("/rules")
def get_rules():
    return {
        "hard_rules": {
            "cgpa_minimum"             : HARD_RULES.get('cgpa_min', 6.5),
            "skills_score_minimum"     : HARD_RULES.get('skills_score_min', 6),
            "programming_languages_min": HARD_RULES.get('programming_langs_min', 2),
            "soft_skills_minimum"      : HARD_RULES.get('soft_skills_min', 4.0),
            "practical_exposure_rule"  : "Must have internship OR >= 0.5 yrs experience",
        }
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # ---- Validate file type ----
    filename = file.filename or ""
    if not (filename.lower().endswith('.pdf') or
            filename.lower().endswith('.docx')):
        raise HTTPException(
            status_code=400,
            detail="Only .pdf and .docx files are supported.")

    # ---- Read file ----
    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.")

    # ---- Parse resume ----
    try:
        parsed = parse_resume(file_bytes, filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Parser error: {str(e)}")

    raw_features = parsed['extracted_features']

    # ---- Hard rules check ----
    passed_rules, rule_message = check_hard_rules(raw_features)

    if not passed_rules:
        return {
            "status"             : "REJECTED",
            "probability"        : 0.0,
            "threshold_used"     : round(THRESHOLD, 4),
            "passed_hard_rules"  : False,
            "reason"             : "Hard rule disqualification",
            "rule_message"       : rule_message,
            "extracted_features" : raw_features,
            "engineered_features": {},
            "explanation"        : {
                "positive_drivers": [],
                "negative_drivers": [
                    {
                        "feature": "Hard Rule Violation",
                        "impact" : -1.0,
                        "detail" : rule_message,
                    }
                ],
            },
            "raw_text_preview"   : parsed['raw_text_preview'],
            "word_count"         : parsed['word_count'],
        }

    # ---- Engineer features ----
    engineered = engineer_features(raw_features)
    df_row     = build_feature_row(engineered)

    # ---- Predict ----
    raw_prob = float(model.predict(df_row)[0])
    prob     = float(np.clip(raw_prob, 0.08, 0.92))

    decision = "SHORTLISTED" if prob >= THRESHOLD else "REJECTED"

    # ---- SHAP explanation ----
    explanation = get_shap_explanation(df_row)

    return {
        "status"             : decision,
        "probability"        : round(prob * 100, 2),
        "threshold_used"     : round(THRESHOLD, 4),
        "passed_hard_rules"  : True,
        "rule_message"       : rule_message,
        "extracted_features" : raw_features,
        "engineered_features": {
            "academic_strength" : round(engineered['academic_strength'], 4),
            "practical_exposure": round(engineered['practical_exposure'], 4),
            "career_momentum"   : round(engineered['career_momentum'], 4),
            "achievement_score" : round(engineered['achievement_score'], 4),
            "technical_breadth" : round(engineered['technical_breadth'], 4),
        },
        "explanation"        : explanation,
        "raw_text_preview"   : parsed['raw_text_preview'],
        "word_count"         : parsed['word_count'],
    }