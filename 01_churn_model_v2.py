"""
Train churn model on insurance_policyholder_churn_synthetic.csv
Outputs: churn_model.pkl, churn_columns.pkl (list of feature names after encoding)
Run from the Backend directory: python 01_churn_model_v2.py
"""
import pathlib, joblib, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA = pathlib.Path(__file__).parent.parent / "extracted_ml" / "insurance_policyholder_churn_synthetic.csv"

df = pd.read_csv(DATA)

DROP = ["customer_id", "as_of_date", "churn_type", "churn_probability_true"]
TARGET = "churn_flag"

df = df.drop(columns=[c for c in DROP if c in df.columns])

X = df.drop(columns=[TARGET])
y = df[TARGET]

CAT_COLS = ["region_name", "age_band", "marital_status", "policy_type", "payment_frequency"]
NUM_COLS = [c for c in X.columns if c not in CAT_COLS]

cat_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
    ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])
num_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
])

pre = ColumnTransformer([
    ("cat", cat_pipe, CAT_COLS),
    ("num", num_pipe, NUM_COLS),
])

model = Pipeline([
    ("pre", pre),
    ("clf", XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )),
])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model.fit(X_train, y_train)

auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
print(f"Validation AUC: {auc:.4f}")

joblib.dump(model, "churn_model.pkl")
joblib.dump(list(X.columns), "churn_columns.pkl")
print("Saved churn_model.pkl and churn_columns.pkl")
print("Feature columns:", list(X.columns))
