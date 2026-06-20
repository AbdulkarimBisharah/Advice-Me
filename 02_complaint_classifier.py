"""
Train complaint classifier on complaints.csv (CFPB dataset).
Classifies narrative text into Issue category.
Outputs: complaint_vectorizer.pkl, complaint_model.pkl, complaint_labels.pkl
Run from the Backend directory: python 02_complaint_classifier.py
"""
import pathlib, joblib, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

DATA = pathlib.Path(__file__).parent.parent / "extracted_nlp" / "complaints.csv"

# Top-N issues to keep — keeps classes balanced enough for a demo
TOP_N = 8
SAMPLE_PER_CLASS = 5000

df = pd.read_csv(DATA, usecols=["narrative", "Issue"], low_memory=False)
df = df.dropna(subset=["narrative", "Issue"])
df["narrative"] = df["narrative"].astype(str)
df["Issue"] = df["Issue"].astype(str)

# Keep only top-N most common issues
top_issues = df["Issue"].value_counts().head(TOP_N).index.tolist()
df = df[df["Issue"].isin(top_issues)]

# Balanced subsample so training is fast
df = pd.concat(
    [g.sample(min(len(g), SAMPLE_PER_CLASS), random_state=42) for _, g in df.groupby("Issue")],
    ignore_index=True,
)

print(f"Training on {len(df)} samples across {df['Issue'].nunique()} classes")

X = df["narrative"]
y = df["Issue"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

vec = TfidfVectorizer(max_features=30_000, ngram_range=(1, 2), sublinear_tf=True)
X_tr_vec = vec.fit_transform(X_train)
X_val_vec = vec.transform(X_val)

clf = LogisticRegression(max_iter=1000, C=5, solver="lbfgs", n_jobs=-1)
clf.fit(X_tr_vec, y_train)

print(classification_report(y_val, clf.predict(X_val_vec), zero_division=0))

joblib.dump(vec, "complaint_vectorizer.pkl")
joblib.dump(clf, "complaint_model.pkl")
joblib.dump(top_issues, "complaint_labels.pkl")
print("Saved complaint_vectorizer.pkl, complaint_model.pkl, complaint_labels.pkl")
