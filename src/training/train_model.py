import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from pathlib import Path

# train_model.py
data = pd.concat([
    pd.read_csv("data/raw/I.csv"),
    pd.read_csv("data/raw/R.csv"),
    pd.read_csv("data/raw/S.csv")
])

X = data.drop("label", axis=1)
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Guardar
model_path = Path("src/training/trained_model.pkl")
model_path.parent.mkdir(parents=True, exist_ok=True)
with open(model_path, "wb") as f:
    pickle.dump(clf, f)
print(f"Modelo guardado en {model_path}")
