import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import pickle

# LOAD DATA
data = pd.read_csv("credit_risk_dataset_with_expenses.csv")

# SELECT FEATURES
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

# TARGET
y = data["loan_status"]

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# PIPELINE (HANDLE NaN + SCALE + MODEL)
model = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("rf", RandomForestClassifier(n_estimators=100))
])

# TRAIN
model.fit(X_train, y_train)

# TEST
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("✅ Model trained successfully")
print("🔥 Accuracy:", accuracy)

# SAVE MODEL
with open("credit_model.pkl", "wb") as f:
    pickle.dump(model, f)