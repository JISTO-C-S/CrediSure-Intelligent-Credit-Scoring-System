import os
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request


app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_data.pkl")
CREDIT_MODEL_PATH = os.path.join(MODELS_DIR, "credit_model.pkl")


ADVICE_DICT = {
    "credit_amount": "Apply for a smaller loan amount.",
    "credit_history": "Improve your credit repayment consistency.",
    "savings_status": "Increase your savings balance over the next 3-6 months.",
    "employment_duration": "Maintain stable employment for a longer period.",
    "existing_credits": "Reduce outstanding debts and liabilities.",
    "duration_months": "Choose a shorter repayment duration.",
    "checking_status": "Improve the balance in your checking account.",
    "installment_rate": "Lower your installment burden relative to income.",
    "age": "Build a longer credit history over time.",
    "purpose": "Re-evaluate the loan purpose or provide stronger collateral.",
    "property": "Improve asset/collateral quality if possible.",
    "residence_duration": "Maintain a stable residence for longer.",
    "housing": "A more stable housing profile can help your score.",
}
DEFAULT_ADVICE = "Improve overall financial stability and credit behavior."

FRIENDLY_NAMES = {
    "checking_status": "Checking account status",
    "duration_months": "Loan duration (months)",
    "credit_history": "Credit history",
    "purpose": "Loan purpose",
    "credit_amount": "Loan amount requested",
    "savings_status": "Savings status",
    "employment_duration": "Employment duration",
    "installment_rate": "Installment burden (1-4)",
    "personal_status_sex": "Personal status",
    "other_debtors": "Other debtors/guarantors",
    "residence_duration": "Residence duration (years)",
    "property": "Property/assets",
    "age": "Age",
    "other_installment_plans": "Other installment plans",
    "housing": "Housing status",
    "existing_credits": "Existing credits",
    "job": "Job level",
    "num_dependents": "Dependents",
    "telephone": "Telephone",
    "foreign_worker": "Foreign worker",
}

NUMERIC_HINTS = {
    "duration_months": "Example: 12, 24, 36",
    "credit_amount": "Example: 5000",
    "installment_rate": "1 to 4 (higher is heavier)",
    "residence_duration": "Years at current address",
    "age": "Years",
    "existing_credits": "Number of active credits",
    "num_dependents": "People depending on your income",
}

CATEGORY_DESCRIPTIONS = {
    "checking_status": {"A11": "Balance < 0", "A12": "Balance 0-200", "A13": "Balance >= 200", "A14": "No checking account"},
    "credit_history": {
        "A30": "No previous credits / all paid on time",
        "A31": "All previous credits paid on time",
        "A32": "Current credits paid on time so far",
        "A33": "Past delays in payment",
        "A34": "Critical account / serious past issues",
    },
    "purpose": {
        "A40": "Car (new)",
        "A41": "Car (used)",
        "A410": "Other purpose",
        "A42": "Furniture/equipment",
        "A43": "Radio/TV",
        "A44": "Domestic appliances",
        "A45": "Repairs",
        "A46": "Education",
        "A48": "Retraining",
        "A49": "Business",
    },
    "savings_status": {
        "A61": "Savings < 100",
        "A62": "Savings 100-500",
        "A63": "Savings 500-1000",
        "A64": "Savings >= 1000",
        "A65": "No savings / unknown",
    },
    "employment_duration": {"A71": "Unemployed", "A72": "< 1 year", "A73": "1-4 years", "A74": "4-7 years", "A75": ">= 7 years"},
    "personal_status_sex": {
        "A91": "Male, divorced/separated",
        "A92": "Female, divorced/separated/married",
        "A93": "Male, single",
        "A94": "Male, married/widowed",
    },
    "other_debtors": {"A101": "None", "A102": "Co-applicant", "A103": "Guarantor"},
    "property": {"A121": "Real estate", "A122": "Savings/life insurance", "A123": "Car/other assets", "A124": "No major property"},
    "other_installment_plans": {"A141": "Bank", "A142": "Store", "A143": "None"},
    "housing": {"A151": "Rent", "A152": "Own home", "A153": "Living for free"},
    "job": {"A171": "Unskilled (non-resident)", "A172": "Unskilled (resident)", "A173": "Skilled employee", "A174": "Management/highly skilled"},
    "telephone": {"A191": "No registered telephone", "A192": "Has registered telephone"},
    "foreign_worker": {"A201": "Yes", "A202": "No"},
}


def friendly_name(feature):
    return FRIENDLY_NAMES.get(feature, feature.replace("_", " ").title())


@lru_cache(maxsize=1)
def load_artifacts():
    if not os.path.exists(PROCESSED_DATA_PATH) or not os.path.exists(CREDIT_MODEL_PATH):
        raise FileNotFoundError("Missing model artifacts. Run data preparation/training first.")
    model = joblib.load(CREDIT_MODEL_PATH)
    artifacts = joblib.load(PROCESSED_DATA_PATH)
    return model, artifacts


def build_form_schema(artifacts):
    rows = []
    num_cols = set(artifacts["numerical_cols"])
    cat_cols = set(artifacts["categorical_cols"])
    for feature in artifacts["feature_names"]:
        row = {
            "name": feature,
            "label": friendly_name(feature),
            "type": "number" if feature in num_cols else "select",
            "hint": NUMERIC_HINTS.get(feature, "") if feature in num_cols else "",
        }
        if feature in cat_cols:
            classes = artifacts["label_encoders"][feature].classes_
            row["options"] = [
                {"value": str(i), "label": CATEGORY_DESCRIPTIONS.get(feature, {}).get(code, code)}
                for i, code in enumerate(classes)
            ]
        rows.append(row)
    return rows


def user_value_text(feature, scaled_df, artifacts):
    val = float(scaled_df.iloc[0][feature])
    if feature in artifacts["numerical_cols"]:
        idx = list(artifacts["numerical_cols"]).index(feature)
        scaler = artifacts["scaler"]
        raw = val * float(scaler.scale_[idx]) + float(scaler.mean_[idx])
        return f"{raw:.0f}"
    if feature in artifacts["categorical_cols"]:
        code = int(round(val))
        classes = artifacts["label_encoders"][feature].classes_
        if 0 <= code < len(classes):
            source = classes[code]
            desc = CATEGORY_DESCRIPTIONS.get(feature, {}).get(source, source)
            return desc
    return str(val)


def suggest_improved_profile(scaled_df):
    improved = scaled_df.copy()
    for col in ["duration_months", "credit_amount", "installment_rate", "existing_credits"]:
        if col in improved.columns:
            improved[col] = np.minimum(improved[col].values, 0.0)
    better_values = {
        "checking_status": 2,
        "savings_status": 3,
        "employment_duration": 4,
        "housing": 1,
        "telephone": 1,
    }
    for col, val in better_values.items():
        if col in improved.columns:
            improved[col] = val
    return improved


def parse_input_to_scaled_df(form_data, artifacts):
    raw = {}
    errors = []
    for feature in artifacts["feature_names"]:
        value = (form_data.get(feature) or "").strip()
        if value == "":
            errors.append(f"Missing value for {friendly_name(feature)}.")
            continue
        try:
            if feature in artifacts["numerical_cols"]:
                raw[feature] = [float(value)]
            else:
                raw[feature] = [int(value)]
        except ValueError:
            errors.append(f"Invalid value for {friendly_name(feature)}.")

    if errors:
        return None, errors

    df = pd.DataFrame(raw)[artifacts["feature_names"]]
    num_cols = list(artifacts["numerical_cols"])
    df[num_cols] = artifacts["scaler"].transform(df[num_cols])
    return df, []


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    errors = []
    form_values = {}
    form_schema = []

    try:
        model, artifacts = load_artifacts()
        form_schema = build_form_schema(artifacts)
    except Exception as exc:  # pragma: no cover
        errors = [str(exc)]
        return render_template("index.html", errors=errors, result=result, form_schema=form_schema, form_values=form_values)

    if request.method == "POST":
        form_values = request.form.to_dict(flat=True)
        scaled_df, parse_errors = parse_input_to_scaled_df(request.form, artifacts)
        if parse_errors:
            errors.extend(parse_errors)
        else:
            risk = float(model.predict_proba(scaled_df)[0][1] * 100)
            approved = risk <= 50
            result = {
                "risk_score": risk,
                "approved": approved,
                "threshold_gap": abs(risk - 50),
                "reasons": [],
                "advice": [],
                "simulation": None,
            }

            if not approved and hasattr(model, "coef_"):
                coefs = model.coef_[0]
                impacts = coefs * scaled_df.iloc[0].values
                impact_df = pd.DataFrame({"feature": artifacts["feature_names"], "impact": impacts})
                top = impact_df[impact_df["impact"] > 0].sort_values("impact", ascending=False).head(3)
                if top.empty:
                    top = impact_df.sort_values("impact", ascending=False).head(3)
                for _, row in top.iterrows():
                    feature = row["feature"]
                    result["reasons"].append(
                        {
                            "feature": friendly_name(feature),
                            "value": user_value_text(feature, scaled_df, artifacts),
                        }
                    )
                    result["advice"].append(ADVICE_DICT.get(feature, DEFAULT_ADVICE))

                improved = suggest_improved_profile(scaled_df)
                improved_risk = float(model.predict_proba(improved)[0][1] * 100)
                result["simulation"] = {
                    "risk_score": improved_risk,
                    "approved": improved_risk <= 50,
                }

    return render_template(
        "index.html",
        errors=errors,
        result=result,
        form_schema=form_schema,
        form_values=form_values,
    )


if __name__ == "__main__":
    app.run(debug=True)
