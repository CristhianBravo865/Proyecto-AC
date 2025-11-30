# src/training/train_model.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

data_dir = Path("data/raw")
model_dir = Path("models")
model_dir.mkdir(parents=True, exist_ok=True)

all_files = list(data_dir.glob("*.csv"))
frames = [pd.read_csv(f) for f in all_files if f.is_file()]
if not frames:
    raise SystemExit("No data found in data/raw. Run capture_letter first.")

df = pd.concat(frames, ignore_index=True)
labels = df["label"].astype(str).values
X = df.drop(columns=["label"]).values
le = LabelEncoder()
y = le.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred, target_names=le.classes_))

joblib.dump(clf, model_dir / "letters.pkl")
joblib.dump(le, model_dir / "label_encoder.pkl")
print("Saved model to models/letters.pkl and encoder to models/label_encoder.pkl")
