import os
import joblib
import pandas as pd

from sklearn.metrics import (
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = "ml/data/loan_approval_data.csv"
MODEL_DIR = "ml/models"
OUTPUT_DIR = "ml/outputs"
TARGET_COL = "loan_status"
RANDOM_STATE = 42


# -----------------------------
# LOAD & PREPROCESS
# -----------------------------
def load_and_preprocess():
    df = pd.read_csv(DATA_PATH)

    # clean column names
    df.columns = df.columns.str.strip()

    # drop id
    df = df.drop("loan_id", axis=1)

    # strip whitespace from string columns (fixes values like ' Approved')
    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].apply(lambda col: col.str.strip())

    # capture original target values for diagnostics
    orig_target_values = df[TARGET_COL].unique()

    # encode categoricals
    df["education"] = df["education"].map({"Graduate": 1, "Not Graduate": 0})

    df["self_employed"] = df["self_employed"].map({"Yes": 1, "No": 0})

    # encode target
    df[TARGET_COL] = df[TARGET_COL].map({"Approved": 1, "Rejected": 0})

    # drop rows with missing target values (prevents stratify errors)
    df = df.dropna(subset=[TARGET_COL])

    # if encoding removed all rows, raise informative error
    if df.shape[0] == 0:
        raise ValueError(
            f"No rows remain after encoding target column '{TARGET_COL}'. "
            f"Original target values found: {list(orig_target_values)}. "
            "Ensure the CSV uses 'Approved'/'Rejected' for the target or update the mapping."
        )

    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    return X, y


# -----------------------------
# MODEL EVALUATION
# -----------------------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # some models may not expose predict_proba
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = None

    metrics = {
        "recall_rejected": recall_score(y_test, y_pred, pos_label=0),
        "precision_rejected": precision_score(y_test, y_pred, pos_label=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_prob)

    return metrics, y_pred, y_prob


# -----------------------------
# FAILURE ANALYSIS
# -----------------------------
def extract_dangerous_failures(X_test, y_test, y_pred, y_prob):
    """
    Dangerous failure:
    Model APPROVES (1) a loan that should be REJECTED (0)
    """
    failures = X_test.copy()
    failures["true_label"] = y_test.values
    failures["predicted_label"] = y_pred

    if y_prob is not None:
        failures["approval_probability"] = y_prob

    dangerous = failures[
        (failures["true_label"] == 0) & (failures["predicted_label"] == 1)
    ]

    # ensure we operate on a copy to avoid SettingWithCopyWarning
    dangerous = dangerous.copy()

    if y_prob is not None:
        dangerous["severity"] = dangerous["approval_probability"]

    return dangerous


# -----------------------------
# MAIN
# -----------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X, y = load_and_preprocess()

    # MUST match train.py
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "Decision_Tree": joblib.load(os.path.join(MODEL_DIR, "decision_tree.pkl")),
        "Random_Forest": joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl")),
        "Gradient_Boosting": joblib.load(
            os.path.join(MODEL_DIR, "gradient_boosting.pkl")
        ),
    }

    all_failures = {}

    print("\n=== LOAN APPROVAL MODEL EVALUATION ===\n")

    for name, model in models.items():
        metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test)
        failures = extract_dangerous_failures(X_test, y_test, y_pred, y_prob)

        all_failures[name] = failures

        # save failures
        failures.to_csv(
            os.path.join(OUTPUT_DIR, f"{name.lower()}_dangerous_approvals.csv"),
            index=True,
        )

        print(f"Model: {name}")
        print(f"Recall (Rejected loans): {metrics['recall_rejected']:.4f}")
        print(f"Precision (Rejected loans): {metrics['precision_rejected']:.4f}")
        print("Confusion Matrix:")
        print(metrics["confusion_matrix"])
        print(f"Dangerous Approvals: {len(failures)}")
        print("-" * 45)

    # -----------------------------
    # COMMON FAILURES
    # -----------------------------
    print("\n=== COMMON DANGEROUS FAILURES ===\n")

    failure_indices = [set(df.index) for df in all_failures.values()]
    common_failures = set.intersection(*failure_indices)

    print(f"Loans wrongly approved by ALL models: {len(common_failures)}")

    if common_failures:
        common_df = X_test.loc[list(common_failures)]
        common_df["true_label"] = y_test.loc[list(common_failures)]
        common_df.to_csv(
            os.path.join(OUTPUT_DIR, "common_dangerous_approvals.csv"), index=True
        )

    print("\n✅ Evaluation complete. Check ml/outputs/ for results.")


# -----------------------------
if __name__ == "__main__":
    main()
