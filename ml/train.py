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

    # drop non-predictive column
    df = df.drop("loan_id", axis=1)

    # encode categorical features
    df["education"] = df["education"].map({
        "Graduate": 1,
        "Not Graduate": 0
    })

    df["self_employed"] = df["self_employed"].map({
        "Yes": 1,
        "No": 0
    })

    # encode target
    df[TARGET_COL] = df[TARGET_COL].map({
        "Approved": 1,
        "Rejected": 0
    })

    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    return X, y


# -----------------------------
# FAST MODEL CONFIGS (DEPLOYMENT)
# -----------------------------
def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(
        max_depth=5,               # reduced
        min_samples_leaf=50,       # increased
        class_weight="balanced",
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=50,            # ↓ from 200
        max_depth=6,
        min_samples_leaf=50,
        class_weight="balanced",
        n_jobs=1,                   # IMPORTANT for Streamlit Cloud
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train):
    model = GradientBoostingClassifier(
        n_estimators=50,            # ↓ from 150
        learning_rate=0.1,
        max_depth=3,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    return model


# -----------------------------
# MAIN
# -----------------------------
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    X, y = load_and_preprocess()

    # SAME split everywhere
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\n⚙️ Training FAST demo models (Streamlit optimized)...\n")

    # Decision Tree
    dt = train_decision_tree(X_train, y_train)
    joblib.dump(dt, os.path.join(MODEL_DIR, "decision_tree.pkl"))
    print("Decision Tree trained | Recall (Rejected):",
          recall_score(y_test, dt.predict(X_test), pos_label=0))

    # Random Forest
    rf = train_random_forest(X_train, y_train)
    joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest.pkl"))
    print("Random Forest trained | Recall (Rejected):",
          recall_score(y_test, rf.predict(X_test), pos_label=0))

    # Gradient Boosting
    gb = train_gradient_boosting(X_train, y_train)
    joblib.dump(gb, os.path.join(MODEL_DIR, "gradient_boosting.pkl"))
    print("Gradient Boosting trained | Recall (Rejected):",
          recall_score(y_test, gb.predict(X_test), pos_label=0))

    print("\n✅ Models trained quickly and saved for Streamlit demo.")


# -----------------------------
if __name__ == "__main__":
    main()
