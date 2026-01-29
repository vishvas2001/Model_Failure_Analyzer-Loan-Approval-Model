# 🏦 Explainable Loan Approval System  
### Tree-Based Model Comparison & Failure Analysis

An **end-to-end Machine Learning project** that goes beyond accuracy and focuses on **risk-aware decision making** in loan approval systems.

Instead of only predicting *Approved / Rejected*, this project analyzes **where ML models fail**, especially **dangerous approvals** that can cause financial loss.

---

## 🚀 Project Motivation

In real banking systems:
- Approving a risky applicant is **far more costly** than rejecting a safe one.
- Most ML projects optimize accuracy and ignore **failure impact**.

This project is built to answer:
> **“Where do ML models make risky loan approval decisions, and why?”**

---

## 🧠 Core Ideas

- Compare **Decision Tree**, **Random Forest**, and **Gradient Boosting**
- Focus on **high-risk failures** instead of leaderboard metrics
- Build an **explainable, user-friendly UI** for real-world understanding
- Apply **business-aligned evaluation**, not academic ML

---

## 📂 Project Structure

```

Loan_Approval_System/
│
├── streamlit_app/
│ ├── app.py # Interactive Streamlit UI
│ └── requirements.txt
│
├── ml/
│ ├── data/
│ │ └── loan_approval_data.csv
│ │
│ ├── models/
│ │ ├── decision_tree.pkl
│ │ ├── random_forest.pkl
│ │ └── gradient_boosting.pkl
│ │
│ ├── outputs/
│ │ ├── decision_tree_dangerous_approvals.csv
│ │ ├── random_forest_dangerous_approvals.csv
│ │ ├── gradient_boosting_dangerous_approvals.csv
│ │ └── common_dangerous_approvals.csv
│ │
│ ├── train.py # Model training
│ └── evaluate.py # Failure-focused evaluation
│
├── notebooks/
│ └── eda.ipynb # Exploratory Data Analysis
│
└── README.md

```

---

## 📊 Dataset Overview

**Loan Approval Dataset** with human-readable, business-relevant features:

- Applicant income
- Loan amount & term
- CIBIL score
- Number of dependents
- Education & employment
- Asset values (bank, residential, commercial, luxury)

**Target:**
- `Approved` → 1  
- `Rejected` → 0  

---

## 🎯 Business-Aligned Failure Definition

### 🔴 Dangerous Failure (Primary Focus)
> **Loan Approved when it should have been Rejected**

This represents:
- High financial risk
- Potential loan default
- Real-world banking loss

The entire evaluation pipeline is designed around minimizing this error.

---

## 📐 Evaluation Metrics

Instead of accuracy, the project prioritizes:

- **Recall (Rejected Loans)** → primary metric  
- **Precision (Rejected Loans)**  
- **Confusion Matrix**  
- **Cross-model failure overlap**

Why?
> Because catching risky applicants matters more than overall accuracy.

---

## 🤖 Models Used

| Model | Purpose |
|----|----|
| Decision Tree | Interpretability & baseline |
| Random Forest | Variance reduction |
| Gradient Boosting | Bias reduction & error correction |

All models use:
- Same train-test split
- Balanced class handling
- Identical evaluation logic

---

## 🧪 Failure Analysis Highlights

- Extracts **dangerous approvals** for each model
- Identifies **common failures** missed by all models
- Analyzes **severity of mistakes**
- Demonstrates that some cases are **inherently hard**

This is **real ML debugging**, not just training.

---

## 🌐 Streamlit Application

The Streamlit app allows users to:

- Enter **realistic loan applicant details**
- Choose ML model (DT / RF / GB)
- View:
  - Approval probability
  - Risk level
  - Decision explanation
- Explore **high-risk demo cases**
- See **why** a decision was made

### UI Design Principles
- No confusing feature names
- Human-readable explanations
- Risk-first messaging
- Manual review warnings for low-confidence cases

---

## ▶️ How to Run the Project

### 1️⃣ Install dependencies
```bash
pip install -r streamlit_app/requirements.txt
```
### 2️⃣ Train models
```
python ml/train.py
```
### 3️⃣ Run evaluation & failure analysis
```
python ml/evaluate.py
```
### 4️⃣ Launch Streamlit app
```
streamlit run streamlit_app/app.py
```
---
## 🧠 Key Takeaways

* Accuracy alone is not enough in high-risk systems

* Failure analysis reveals hidden weaknesses

* Explainability builds user trust

* Tree-based models behave very differently under risk
---
## 🏆 Why This Project Stands Out

✔ Focuses on failure impact, not just metrics
✔ Business-aligned ML thinking
✔ Explainable and demo-ready
✔ Real-world decision support mindset

This project reflects how ML is actually used in production systems.

---

## 👤 Author

**Vishvas Parmar**
Aspiring Machine Learning Engineer | Data Science Enthusiast

“Understanding why models fail is more valuable than just making them accurate.”

---