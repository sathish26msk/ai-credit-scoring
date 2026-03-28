import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os

app = Flask(__name__)

# =========================
# LOAD DATA & TRAIN MODEL
# =========================
data = pd.read_csv("credit_risk_dataset_with_expenses.csv")

X = data[
    [
        "person_age",
        "person_income",
        "person_emp_length",
        "loan_amnt",
        "monthly_expense",
        "cb_person_cred_hist_length"
    ]
]

y = data["loan_status"]

model = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=1000))
])

model.fit(X, y)

# =========================
# HOME PAGE
# =========================
@app.route('/')
def home():
    return render_template('index.html')


# =========================
# PREDICTION
# =========================
@app.route('/predict', methods=['POST'])
def predict():

    age = int(request.form['age'])
    income = int(request.form['monthly_income'])
    employment = int(request.form['employment_years'])
    loan = int(request.form['existing_loan'])
    expense = int(request.form['monthly_expense'])

    history_input = request.form['payment_history']

    # SIMPLE mapping (safe)
    if history_input == "2":
        history = 5
    elif history_input == "1":
        history = 3
    else:
        history = 1

    features = np.array([[ 
        age, income, employment, loan, expense, history
    ]])

    # =========================
    # AI PROBABILITY
    # =========================
    prob = model.predict_proba(features)[0][1]
    risk_percent = round(prob * 100, 2)

    # =========================
    # BETTER THRESHOLD (IMPORTANT)
    # =========================
    if prob > 0.75:
        result = "High Risk"
        color = "red"
    elif prob > 0.40:
        result = "Medium Risk"
        color = "orange"
    else:
        result = "Low Risk"
        color = "green"

    # =========================
    # SIMPLE REASON SYSTEM
    # =========================
    reasons = []

    if loan > income * 0.6:
        reasons.append("Loan amount is high compared to income")

    if expense > income * 0.5:
        reasons.append("Monthly expenses are high")

    if employment < 2:
        reasons.append("Low employment stability")

    if history < 3:
        reasons.append("Poor credit history")

    return render_template(
        "result.html",
        result=result,
        risk=risk_percent,
        color=color,
        reasons=reasons
    )


# =========================
# RUN APP
# =========================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
