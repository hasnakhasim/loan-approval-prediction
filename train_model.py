
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# 1. Load the dataset
data = pd.read_csv("loan_data.csv")
data.columns = data.columns.str.strip()

# 2. Check the required columns
target = "Loan_Status"

if target not in data.columns:
    raise ValueError(
        "Loan_Status column not found. "
        "Use the labelled training CSV, not test.csv."
    )

# Remove rows without a known target
data = data.dropna(subset=[target]).copy()

# Standardize target labels
data[target] = data[target].astype(str).str.strip().str.upper()
data = data[data[target].isin(["Y", "N"])]

if data.empty:
    raise ValueError("No valid Y/N values found in Loan_Status.")

# Remove identifier column if present
data = data.drop(columns=["Loan_ID"], errors="ignore")

X = data.drop(columns=[target])
y = data[target]

# Treat these as categorical fields, even if stored as numbers
categorical_candidates = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Credit_History",
    "Property_Area"
]

categorical_features = [
    col for col in categorical_candidates if col in X.columns
]
numerical_features = [
    col for col in X.columns if col not in categorical_features
]

# Convert numerical columns to numeric values
for col in numerical_features:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# 3. Split the data before fitting preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Fill missing values and encode categories
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# 5. Create the Random Forest model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    ))
])

# 6. Train
model.fit(X_train, y_train)

# 7. Evaluate on the held-out test set
predictions = model.predict(X_test)

print("\n--- Loan Approval Model Evaluation ---")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print(f"Accuracy:  {accuracy_score(y_test, predictions):.3f}")
print(
    f"Precision (approved): "
    f"{precision_score(y_test, predictions, pos_label='Y', zero_division=0):.3f}"
)
print(
    f"Recall (approved): "
    f"{recall_score(y_test, predictions, pos_label='Y', zero_division=0):.3f}"
)
print(
    f"F1-score (approved): "
    f"{f1_score(y_test, predictions, pos_label='Y', zero_division=0):.3f}"
)

print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

print("Confusion matrix (label order: N, Y):")
print(confusion_matrix(y_test, predictions, labels=["N", "Y"]))

# 8. Save the complete pipeline
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/loan_model.pkl")

print("\nModel saved to model/loan_model.pkl")
