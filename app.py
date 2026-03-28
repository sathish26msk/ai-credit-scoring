import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os

app = Flask(__name__)

# 🔥 LOAD DATA
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

# 🔥 MODEL PIPELINE
model = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

# 🔥 TRAIN MODEL
model.fit(X, y)

# -------------------------

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    age = int(request.form['age'])
    income = int(request.form['monthly_income'])
    employment = int(request.form['employment_years'])
    loan = int(request.form['existing_loan'])
    expense = int(request.form['monthly_expense'])

    history_input = request.form['payment_history']

    # 🔥 Convert user input
    if history_input == "2":      # Good
        history = 5
    elif history_input == "1":    # Average
        history = 3
    else:                         # Poor
        history = 1

    # 🔥 Create feature array
    features = np.array([[ 
        age, income, employment, loan, expense, history
    ]])

    # 🔥 Predict probability
    prob = model.predict_proba(features)[0][1]
    risk_percent = round(prob * 100, 2)

    print("Probability:", prob)   # Debug (optional)

    # 🔥 BALANCED THRESHOLD (FIXED)
    if prob > 0.7:
        result = "High Risk"
    elif prob > 0.4:
        result = "Medium Risk"
    else:
        result = "Low Risk"

    return render_template("result.html", result=result, risk=risk_percent)


# -------------------------

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
