import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os

app = Flask(__name__)

# ========================
# LOAD DATA & TRAIN MODEL
# ========================
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

# ========================
# HOME PAGE
# ========================
@app.route('/')
def home():
    return render_template('index.html')


# ========================
# PREDICT
# ========================
@app.route('/predict', methods=['POST'])
def predict():

    age = int(request.form['age'])
    income = int(request.form['monthly_income'])
    employment = int(request.form['employment_years'])
    loan = int(request.form['existing_loan'])
    expense = int(request.form['monthly_expense'])

    history_input = request.form['payment_history']

    if history_input == "2":
        history = 5
    elif history_input == "1":
        history = 3
    else:
        history = 1

    features = np.array([[age, income, employment, loan, expense, history]])

    prob = model.predict_proba(features)[0][1]
    risk_percent = round(prob * 100, 2)

    # ========================
    # RISK CATEGORY
    # ========================
    if prob > 0.75:
        result = "High Risk"
    elif prob > 0.40:
        result = "Medium Risk"
    else:
        result = "Low Risk"

    # ========================
    # EXPLANATION (KEY 🔥)
    # ========================
    reasons = []

    if result == "Low Risk":
        if income > loan * 2:
            reasons.append("Income is strong compared to loan")
        if expense < income * 0.4:
            reasons.append("Expenses are well controlled")
        if employment >= 3:
            reasons.append("Stable employment")
        if history >= 3:
            reasons.append("Good credit history")

    elif result == "Medium Risk":
        if loan > income * 0.3:
            reasons.append("Loan amount is moderate")
        if expense > income * 0.4:
            reasons.append("Expenses are slightly high")
        if employment < 5:
            reasons.append("Moderate job stability")
        reasons.append("Balanced risk between safe and risky factors")

    elif result == "High Risk":
        if loan > income * 0.6:
            reasons.append("Loan is very high compared to income")
        if expense > income * 0.5:
            reasons.append("Expenses are too high")
        if employment < 2:
            reasons.append("Unstable employment")
        if history < 3:
            reasons.append("Poor credit history")

    return render_template(
        "result.html",
        result=result,
        risk=risk_percent,
        reasons=reasons
    )


# ========================
# RUN
# ========================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
