"""
Regenerates the 3 preprocessing artefacts that match the Colab-trained
LightGBM model (churn_model.joblib).

Mirrors the exact steps from hackathon_ml_training_part.ipynb:
  - Drop same columns
  - LabelEncoder per categorical column (fit on full dataset, same as Colab)
  - StandardScaler (fit on train split, same random_state=42)
  - Save feature column list

Run from the Backend directory: python generate_preprocessor.py
"""
import pathlib, joblib, pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

DATA = pathlib.Path(__file__).parent.parent / "extracted_ml" / "insurance_policyholder_churn_synthetic.csv"

df = pd.read_csv(DATA)
df = df.drop(columns=["customer_id", "as_of_date", "churn_type", "churn_probability_true"])

cat_cols = df.select_dtypes(include="object").columns.tolist()

le_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_encoders[col] = le

X = df.drop(columns=["churn_flag"])
y = df["churn_flag"]
feat_cols = list(X.columns)

X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
scaler.fit(X_train)

joblib.dump(scaler,       "churn_preprocessor.joblib")
joblib.dump(le_encoders,  "churn_label_encoders.joblib")
joblib.dump(feat_cols,    "churn_feature_columns.joblib")

print(f"Saved churn_preprocessor.joblib ({len(feat_cols)} features)")
print(f"Saved churn_label_encoders.joblib (cols: {cat_cols})")
print(f"Saved churn_feature_columns.joblib")
print("Feature order:", feat_cols)
