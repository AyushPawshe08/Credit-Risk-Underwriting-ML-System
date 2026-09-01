from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from credit_underwriting.pipeline import (
    generate_rejection_reasons,
    load_model_bundle,
    predict_default_risk,
)


ROOT = Path(__file__).resolve().parent
ARTIFACT_PATH = ROOT / "artifacts" / "model_bundle.joblib"
RISK_THRESHOLD = 0.50

app = FastAPI(
    title="Credit Risk Underwriting API",
    description="Predict loan default risk and generate model-based rejection reason candidates.",
    version="0.1.0",
)


class Applicant(BaseModel):
    person_age: int = Field(..., ge=18, le=100)
    person_income: float = Field(..., gt=0)
    person_home_ownership: Literal["RENT", "OWN", "MORTGAGE", "OTHER"]
    person_emp_length: float | None = Field(None, ge=0)
    loan_intent: Literal[
        "PERSONAL",
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "HOMEIMPROVEMENT",
        "DEBTCONSOLIDATION",
    ]
    loan_grade: Literal["A", "B", "C", "D", "E", "F", "G"]
    loan_amnt: float = Field(..., gt=0)
    loan_int_rate: float | None = Field(None, ge=0)
    loan_percent_income: float = Field(..., ge=0)
    cb_person_default_on_file: Literal["Y", "N"]
    cb_person_cred_hist_length: int = Field(..., ge=0)


class PredictionResponse(BaseModel):
    default_risk: float
    risk_band: Literal["low", "medium", "high"]
    decision: Literal["approve", "review", "reject"]
    rejection_reasons: list[str]
    model_name: str


def get_bundle() -> dict:
    if not ARTIFACT_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Model artifact is missing. Run `python train_pipeline.py` first.",
        )
    return load_model_bundle(ARTIFACT_PATH)


def score_to_band(score: float) -> str:
    if score >= RISK_THRESHOLD:
        return "high"
    if score >= 0.25:
        return "medium"
    return "low"


def score_to_decision(score: float) -> str:
    if score >= RISK_THRESHOLD:
        return "reject"
    if score >= 0.25:
        return "review"
    return "approve"


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "artifact_exists": ARTIFACT_PATH.exists(),
    }


@app.get("/metadata")
def metadata() -> dict:
    bundle = get_bundle()
    benchmark_results = bundle["benchmark_results"].round(4).to_dict(orient="records")
    return {
        "model_name": bundle["best_model_name"],
        "risk_threshold": RISK_THRESHOLD,
        "input_columns": bundle["input_columns"],
        "benchmark_results": benchmark_results,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(applicant: Applicant) -> PredictionResponse:
    bundle = get_bundle()
    applicant_frame = pd.DataFrame([applicant.model_dump()])
    risk_score = float(predict_default_risk(applicant_frame, bundle)[0])

    rejection_reasons = []
    if risk_score >= RISK_THRESHOLD:
        rejection_reasons = generate_rejection_reasons(applicant_frame, bundle, top_n=3)[0]

    return PredictionResponse(
        default_risk=round(risk_score, 6),
        risk_band=score_to_band(risk_score),
        decision=score_to_decision(risk_score),
        rejection_reasons=rejection_reasons,
        model_name=bundle["best_model_name"],
    )
