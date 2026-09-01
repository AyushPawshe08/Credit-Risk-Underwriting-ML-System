from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request

from credit_underwriting.pipeline import (
    generate_rejection_reasons,
    load_model_bundle,
    predict_default_risk,
)


ROOT = Path(__file__).resolve().parent
ARTIFACT_PATH = ROOT / "artifacts" / "model_bundle.joblib"
RISK_THRESHOLD = 0.50
REVIEW_THRESHOLD = 0.25

app = Flask(__name__)
model_bundle = load_model_bundle(ARTIFACT_PATH)


FIELD_OPTIONS = {
    "person_home_ownership": ["RENT", "OWN", "MORTGAGE", "OTHER"],
    "loan_intent": [
        "PERSONAL",
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "HOMEIMPROVEMENT",
        "DEBTCONSOLIDATION",
    ],
    "loan_grade": ["A", "B", "C", "D", "E", "F", "G"],
    "cb_person_default_on_file": ["N", "Y"],
}

DEFAULT_FORM = {
    "person_age": 30,
    "person_income": 65000,
    "person_home_ownership": "RENT",
    "person_emp_length": 5,
    "loan_intent": "DEBTCONSOLIDATION",
    "loan_grade": "B",
    "loan_amnt": 12000,
    "loan_int_rate": 11.5,
    "loan_percent_income": 0.18,
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 6,
}

FEATURE_HELP = {
    "person_age": "Applicant's age in years.",
    "person_income": "Applicant's annual income before tax.",
    "person_home_ownership": "Applicant's housing status: renting, owns home, mortgage, or other.",
    "person_emp_length": "Number of years the applicant has been employed.",
    "loan_intent": "The stated purpose of the loan, such as education, medical, or debt consolidation.",
    "loan_grade": "Credit grade assigned to the loan. A is strongest; G is weakest.",
    "loan_amnt": "Total amount of money the applicant wants to borrow.",
    "loan_int_rate": "Annual interest rate assigned to the loan.",
    "loan_percent_income": "Loan amount divided by annual income. Example: 0.20 means the loan is 20% of income.",
    "cb_person_default_on_file": "Whether the credit bureau file shows a prior default. Y means yes; N means no.",
    "cb_person_cred_hist_length": "Number of years of credit history available for the applicant.",
}


def parse_form(form_data) -> dict:
    parsed = {
        "person_age": int(form_data["person_age"]),
        "person_income": float(form_data["person_income"]),
        "person_home_ownership": form_data["person_home_ownership"],
        "person_emp_length": float(form_data["person_emp_length"]),
        "loan_intent": form_data["loan_intent"],
        "loan_grade": form_data["loan_grade"],
        "loan_amnt": float(form_data["loan_amnt"]),
        "loan_int_rate": float(form_data["loan_int_rate"]),
        "loan_percent_income": float(form_data["loan_percent_income"]),
        "cb_person_default_on_file": form_data["cb_person_default_on_file"],
        "cb_person_cred_hist_length": int(form_data["cb_person_cred_hist_length"]),
    }
    return parsed


def classify_risk(score: float) -> tuple[str, str]:
    if score >= RISK_THRESHOLD:
        return "Reject", "high"
    if score >= REVIEW_THRESHOLD:
        return "Manual Review", "medium"
    return "Approve", "low"


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    errors = []
    form_values = DEFAULT_FORM.copy()

    if request.method == "POST":
        try:
            form_values = parse_form(request.form)
            applicant_frame = pd.DataFrame([form_values])
            risk_score = float(predict_default_risk(applicant_frame, model_bundle)[0])
            decision, risk_band = classify_risk(risk_score)
            reasons = []

            if risk_score >= RISK_THRESHOLD:
                reasons = generate_rejection_reasons(applicant_frame, model_bundle, top_n=3)[0]

            result = {
                "risk_score": risk_score,
                "risk_percent": risk_score * 100,
                "decision": decision,
                "risk_band": risk_band,
                "reasons": reasons,
                "model_name": model_bundle["best_model_name"],
            }
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"Please check the form values. Details: {exc}")

    benchmark_results = model_bundle["benchmark_results"].round(4).to_dict(orient="records")
    return render_template(
        "index.html",
        field_options=FIELD_OPTIONS,
        form_values=form_values,
        result=result,
        errors=errors,
        benchmark_results=benchmark_results,
        feature_help=FEATURE_HELP,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
