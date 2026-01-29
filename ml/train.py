import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import recall_score


# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = "ml/data/loan_approval_data.csv"
MODEL_DIR = "ml/models"
TARGET_COL = "loan_status"
RANDOM_STATE = 42


# -----------------------------
# LOAD & PREPROCESS DATA
# -----------------------------
def load_and_preprocess():
    df = pd.read_csv(DATA_PATH)

    # clean column names
    df.columns = df.columns.str.strip()

    # strip whitespace from string columns (fixes values like ' Approved')
    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].apply(lambda col: col.str.strip())

    # drop non-predictive column
    df = df.drop("loan_id", axis=1)

    # encode categorical features
    df["education"] = df["education"].map({"Graduate": 1, "Not Graduate": 0})

    df["self_employed"] = df["self_employed"].map({"Yes": 1, "No": 0})

    # capture original target values for diagnostics
    orig_target_values = df[TARGET_COL].unique()

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
# MODEL TRAINING
# -----------------------------
def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=40,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=30,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train):
    model = GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    return model


# -----------------------------
# MAIN
# -----------------------------
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    X, y = load_and_preprocess()

    # IMPORTANT: same split everywhere
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    print("\nTraining Loan Approval Models...\n")

    # Decision Tree
    dt = train_decision_tree(X_train, y_train)
    joblib.dump(dt, os.path.join(MODEL_DIR, "decision_tree.pkl"))
    print(
        "Decision Tree Recall (Rejected loans):",
        recall_score(y_test, dt.predict(X_test), pos_label=0),
    )

    # Random Forest
    rf = train_random_forest(X_train, y_train)
    joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest.pkl"))
    print(
        "Random Forest Recall (Rejected loans):",
        recall_score(y_test, rf.predict(X_test), pos_label=0),
    )

    # Gradient Boosting
    gb = train_gradient_boosting(X_train, y_train)
    joblib.dump(gb, os.path.join(MODEL_DIR, "gradient_boosting.pkl"))
    print(
        "Gradient Boosting Recall (Rejected loans):",
        recall_score(y_test, gb.predict(X_test), pos_label=0),
    )

    print("\n✅ All loan approval models trained and saved.")


# -----------------------------
if __name__ == "__main__":
    main()
