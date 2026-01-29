import streamlit as st
import pandas as pd
import joblib
import os
import subprocess

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATHS = {
    "Decision Tree": "ml/models/decision_tree.pkl",
    "Random Forest": "ml/models/random_forest.pkl",
    "Gradient Boosting": "ml/models/gradient_boosting.pkl"
}

st.set_page_config(page_title="Loan Approval System", layout="centered")

# AUTO TRAIN

def ensure_models_exist():
    if not os.path.exists("ml/models"):
        os.makedirs("ml/models")

    required_models = [
        "decision_tree.pkl",
        "random_forest.pkl",
        "gradient_boosting.pkl"
    ]

    models_missing = any(
        not os.path.exists(os.path.join("ml/models", m))
        for m in required_models
    )

    if models_missing:
        st.warning("Models not found. Training models… (first run only)")
        subprocess.run(["python", "ml/train.py"], check=True)



# -----------------------------
# LOAD MODELS
# -----------------------------

def load_models():
    models = {}
    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            st.error(f"❌ Model file not found: {path}")
            st.stop()
        models[name] = joblib.load(path)
    return models


# 🔥 ENSURE MODELS FIRST
ensure_models_exist()

# 🔥 THEN LOAD MODELS (NO CACHE)
models = load_models()



models = load_models()

# -----------------------------
# UI HEADER
# -----------------------------
st.title("🏦 Explainable Loan Approval System")
st.markdown(
    """
This system demonstrates **how ML models evaluate loan applications**  
with a focus on **risky approvals and model uncertainty**, not just accuracy.
"""
)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Model Selection")
model_name = st.sidebar.selectbox("Choose Model", list(models.keys()))
model = models[model_name]

st.sidebar.markdown("---")
st.sidebar.info("⚠️ Focus: Avoiding risky loan approvals")

# -----------------------------
# DEMO BUTTON
# -----------------------------
st.subheader("Applicant Details")

if st.button("⚠️ Load High-Risk Applicant Example"):
    st.session_state.demo = {
        "income_annum": 200000,
        "loan_amount": 3000000,
        "loan_term": 20,
        "cibil_score": 550,
        "no_of_dependents": 3,
        "education": "Not Graduate",
        "self_employed": "Yes",
        "residential_assets_value": 0,
        "commercial_assets_value": 0,
        "luxury_assets_value": 0,
        "bank_asset_value": 50000
    }

demo = st.session_state.get("demo", {})

# -----------------------------
# INPUT FORM
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    income_annum = st.number_input("Annual Income", value=demo.get("income_annum", 500000))
    loan_amount = st.number_input("Loan Amount", value=demo.get("loan_amount", 1500000))
    loan_term = st.number_input("Loan Term (years)", value=demo.get("loan_term", 15))
    cibil_score = st.slider("CIBIL Score", 300, 900, demo.get("cibil_score", 700))

with col2:
    no_of_dependents = st.number_input("Number of Dependents", value=demo.get("no_of_dependents", 1))
    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"],
        index=0 if demo.get("education", "Graduate") == "Graduate" else 1
    )
    self_employed = st.selectbox(
        "Self Employed",
        ["No", "Yes"],
        index=0 if demo.get("self_employed", "No") == "No" else 1
    )

    residential_assets_value = st.number_input(
        "Residential Assets Value", value=demo.get("residential_assets_value", 2000000)
    )
    commercial_assets_value = st.number_input(
        "Commercial Assets Value", value=demo.get("commercial_assets_value", 0)
    )
    luxury_assets_value = st.number_input(
        "Luxury Assets Value", value=demo.get("luxury_assets_value", 0)
    )
    bank_asset_value = st.number_input(
        "Bank Assets Value", value=demo.get("bank_asset_value", 300000)
    )

# -----------------------------
# PREPARE INPUT
# -----------------------------
input_df = pd.DataFrame([{
    "no_of_dependents": no_of_dependents,
    "education": 1 if education == "Graduate" else 0,
    "self_employed": 1 if self_employed == "Yes" else 0,
    "income_annum": income_annum,
    "loan_amount": loan_amount,
    "loan_term": loan_term,
    "cibil_score": cibil_score,
    "residential_assets_value": residential_assets_value,
    "commercial_assets_value": commercial_assets_value,
    "luxury_assets_value": luxury_assets_value,
    "bank_asset_value": bank_asset_value
}])

# -----------------------------
# PREDICTION
# -----------------------------
st.markdown("---")
st.subheader("Model Decision")

if st.button("Evaluate Loan Application"):
    prob = model.predict_proba(input_df)[0][1]
    approved = prob >= 0.5

    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Approval Probability", f"{prob:.2f}")
    c2.metric("Decision", "APPROVED ✅" if approved else "REJECTED ❌")
    c3.metric(
        "Risk Level",
        "HIGH 🚨" if approved and prob < 0.7 else
        "MEDIUM ⚠️" if approved else
        "LOW ✅"
    )

    # Risk bar
    st.markdown("#### Risk Indicator")
    st.progress(int(prob * 100))

    # -----------------------------
    # WHY THIS DECISION?
    # -----------------------------
    st.markdown("### 🔍 Why this decision?")
    reasons = []

    if cibil_score < 650:
        reasons.append("Low CIBIL score indicates weak credit history")
    if loan_amount > income_annum * 4:
        reasons.append("Loan amount is high relative to income")
    if bank_asset_value < loan_amount * 0.1:
        reasons.append("Low liquid assets to support loan")
    if no_of_dependents >= 3:
        reasons.append("High number of dependents increases financial burden")

    if reasons:
        for r in reasons:
            st.warning(r)
    else:
        st.success("Applicant shows financially stable characteristics")

    # Confidence warning
    if 0.45 <= prob <= 0.55:
        st.info(
            "⚠️ Model confidence is low. A real system would recommend manual review."
        )

    st.caption(
        "This system emphasizes **risk-aware ML decisions**, not blind approvals."
    )
