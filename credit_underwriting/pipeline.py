from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


TARGET = "loan_status"
RATIO_COLUMNS = [
    "dti_proxy",
    "interest_burden",
    "income_to_loan_ratio",
    "employment_to_age_ratio",
    "credit_history_to_age_ratio",
    "income_per_credit_year",
]


def load_credit_data(data_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(data_path)


def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )


def mark_impossible_values_as_missing(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data.loc[data["person_age"] > 100, "person_age"] = np.nan
    data.loc[data["person_emp_length"] > 60, "person_emp_length"] = np.nan
    data.loc[data["person_emp_length"] > (data["person_age"] - 14), "person_emp_length"] = np.nan
    return data


def fit_cleaning_rules(X_train: pd.DataFrame) -> dict:
    X_train_marked = mark_impossible_values_as_missing(X_train)
    numeric_columns = X_train_marked.select_dtypes(include="number").columns.tolist()
    return {
        "numeric_columns": numeric_columns,
        "train_medians": X_train_marked[numeric_columns].median(),
        "train_upper_caps": X_train_marked[numeric_columns].quantile(0.995),
    }


def apply_cleaning(data: pd.DataFrame, cleaning_rules: dict) -> pd.DataFrame:
    data = mark_impossible_values_as_missing(data)
    numeric_columns = cleaning_rules["numeric_columns"]
    data[numeric_columns] = data[numeric_columns].fillna(cleaning_rules["train_medians"])
    data[numeric_columns] = data[numeric_columns].clip(
        upper=cleaning_rules["train_upper_caps"],
        axis=1,
    )
    return data


def add_financial_ratios(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    safe_income = data["person_income"].replace(0, np.nan)
    safe_age = data["person_age"].replace(0, np.nan)
    safe_credit_history = data["cb_person_cred_hist_length"].replace(0, np.nan)

    data["dti_proxy"] = data["loan_amnt"] / safe_income
    data["interest_burden"] = (data["loan_amnt"] * data["loan_int_rate"] / 100) / safe_income
    data["income_to_loan_ratio"] = safe_income / data["loan_amnt"].replace(0, np.nan)
    data["employment_to_age_ratio"] = data["person_emp_length"] / safe_age
    data["credit_history_to_age_ratio"] = data["cb_person_cred_hist_length"] / safe_age
    data["income_per_credit_year"] = safe_income / safe_credit_history
    data[RATIO_COLUMNS] = data[RATIO_COLUMNS].replace([np.inf, -np.inf], np.nan)
    return data


def fit_feature_rules(X_train_clean: pd.DataFrame) -> dict:
    X_train_features = add_financial_ratios(X_train_clean)
    return {"train_ratio_medians": X_train_features[RATIO_COLUMNS].median()}


def apply_feature_engineering(data: pd.DataFrame, feature_rules: dict) -> pd.DataFrame:
    data = add_financial_ratios(data)
    data[RATIO_COLUMNS] = data[RATIO_COLUMNS].fillna(feature_rules["train_ratio_medians"])
    return data


def fit_preprocessor(X_train_features: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X_train_features.select_dtypes(include="number").columns.tolist()
    categorical_features = X_train_features.select_dtypes(exclude="number").columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    preprocessor.fit(X_train_features)
    return preprocessor


def build_models(y_train: pd.Series) -> dict:
    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()
    scale_pos_weight = negative_count / positive_count
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
    }


def benchmark_models(models: dict, X_train_prepared, X_test_prepared, y_train, y_test):
    rows = []
    trained_models = {}
    for model_name, model in models.items():
        model.fit(X_train_prepared, y_train)
        y_proba = model.predict_proba(X_test_prepared)[:, 1]
        y_pred = (y_proba >= 0.50).astype(int)
        rows.append(
            {
                "model": model_name,
                "PR-AUC": average_precision_score(y_test, y_proba),
                "ROC-AUC": roc_auc_score(y_test, y_proba),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "accuracy": accuracy_score(y_test, y_pred),
            }
        )
        trained_models[model_name] = model

    results = (
        pd.DataFrame(rows)
        .sort_values(by="PR-AUC", ascending=False)
        .reset_index(drop=True)
    )
    best_model_name = results.loc[0, "model"]
    return results, trained_models, best_model_name


def train_model_bundle(data_path: str | Path) -> dict:
    df = load_credit_data(data_path)
    X_train, X_test, y_train, y_test = split_features_target(df)

    cleaning_rules = fit_cleaning_rules(X_train)
    X_train_clean = apply_cleaning(X_train, cleaning_rules)
    X_test_clean = apply_cleaning(X_test, cleaning_rules)

    feature_rules = fit_feature_rules(X_train_clean)
    X_train_features = apply_feature_engineering(X_train_clean, feature_rules)
    X_test_features = apply_feature_engineering(X_test_clean, feature_rules)

    preprocessor = fit_preprocessor(X_train_features)
    X_train_prepared = preprocessor.transform(X_train_features)
    X_test_prepared = preprocessor.transform(X_test_features)

    models = build_models(y_train)
    benchmark_results, trained_models, best_model_name = benchmark_models(
        models,
        X_train_prepared,
        X_test_prepared,
        y_train,
        y_test,
    )

    return {
        "cleaning_rules": cleaning_rules,
        "feature_rules": feature_rules,
        "preprocessor": preprocessor,
        "best_model_name": best_model_name,
        "best_model": trained_models[best_model_name],
        "trained_models": trained_models,
        "benchmark_results": benchmark_results,
        "input_columns": X_train.columns.tolist(),
        "feature_columns": X_train_features.columns.tolist(),
        "encoded_feature_names": preprocessor.get_feature_names_out(),
    }


def prepare_applicant_data(applicant_data: pd.DataFrame, bundle: dict):
    cleaned = apply_cleaning(applicant_data, bundle["cleaning_rules"])
    featured = apply_feature_engineering(cleaned, bundle["feature_rules"])
    return bundle["preprocessor"].transform(featured)


def predict_default_risk(applicant_data: pd.DataFrame, bundle: dict) -> np.ndarray:
    prepared = prepare_applicant_data(applicant_data, bundle)
    return bundle["best_model"].predict_proba(prepared)[:, 1]


def prepared_to_dataframe(prepared_matrix, feature_names) -> pd.DataFrame:
    if hasattr(prepared_matrix, "toarray"):
        prepared_matrix = prepared_matrix.toarray()
    return pd.DataFrame(prepared_matrix, columns=feature_names)


def rejection_reason_from_feature(feature_name: str) -> str | None:
    clean_name = feature_name.replace("numeric__", "").replace("categorical__", "")

    reason_map = {
        "dti_proxy": "Requested loan amount is high relative to stated income.",
        "loan_percent_income": "Loan payment burden is high relative to stated income.",
        "interest_burden": "Estimated interest burden is high relative to stated income.",
        "income_to_loan_ratio": "Income support for the requested loan amount is limited.",
        "person_income": "Stated income is a material risk factor for this application.",
        "loan_amnt": "Requested loan amount is a material risk factor for this application.",
        "loan_int_rate": "Quoted interest rate indicates elevated credit risk.",
        "person_emp_length": "Employment history length is a material risk factor.",
        "employment_to_age_ratio": "Employment stability is limited relative to applicant profile.",
        "cb_person_cred_hist_length": "Credit history length is a material risk factor.",
        "credit_history_to_age_ratio": "Credit history depth is limited relative to applicant profile.",
        "income_per_credit_year": "Income relative to credit history depth is a material risk factor.",
    }

    if clean_name == "person_age":
        return None
    if clean_name.startswith("loan_grade_"):
        return "Loan grade indicates elevated credit risk."
    if clean_name.startswith("cb_person_default_on_file_Y"):
        return "Prior default history is present in the credit file."
    if clean_name.startswith("person_home_ownership_RENT"):
        return "Housing status indicates limited collateral or ownership stability."
    if clean_name.startswith("person_home_ownership_OTHER"):
        return "Housing status is a material risk factor."
    if clean_name.startswith("loan_intent_"):
        return "Loan purpose is associated with elevated repayment risk."

    return reason_map.get(clean_name, f"{clean_name} is a material risk factor.")


def generate_rejection_reasons(applicant_data: pd.DataFrame, bundle: dict, top_n: int = 3) -> list[list[str]]:
    prepared = prepare_applicant_data(applicant_data, bundle)
    feature_names = bundle["encoded_feature_names"]
    explain_frame = prepared_to_dataframe(prepared, feature_names)

    explainer = shap.TreeExplainer(bundle["best_model"])
    raw_shap_values = explainer.shap_values(explain_frame)

    if isinstance(raw_shap_values, list):
        shap_values_for_risk = raw_shap_values[1]
    elif getattr(raw_shap_values, "ndim", 0) == 3:
        shap_values_for_risk = raw_shap_values[:, :, 1]
    else:
        shap_values_for_risk = raw_shap_values

    applicant_reasons = []
    for applicant_position in range(len(applicant_data)):
        applicant_shap = pd.Series(
            shap_values_for_risk[applicant_position],
            index=feature_names,
        ).sort_values(ascending=False)

        reasons = []
        for feature_name, shap_value in applicant_shap.items():
            if shap_value <= 0:
                continue
            reason = rejection_reason_from_feature(feature_name)
            if reason and reason not in reasons:
                reasons.append(reason)
            if len(reasons) == top_n:
                break
        applicant_reasons.append(reasons)

    return applicant_reasons


def save_model_bundle(bundle: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)


def load_model_bundle(path: str | Path) -> dict:
    return joblib.load(path)
