# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from pathlib import Path
import glob

# ============================
# Cargar automáticamente todos los CSV
# ============================

data_files = glob.glob("data/raw/*.csv")

print("Cargando archivos:")
for f in data_files:
    print(" -", f)

data = pd.concat([pd.read_csv(f) for f in data_files], ignore_index=True)

# ============================
# Preparar datos
# ============================

X = data.drop("label", axis=1)
y = data["label"]

# ============================
# Entrenar modelo
# ============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    random_state=42
)

clf.fit(X_train, y_train)

# ============================
# Evaluación
# ============================

y_pred = clf.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# ============================
# Guardar modelo
# ============================

model_path = Path("../training/trained_model.pkl")
model_path.parent.mkdir(parents=True, exist_ok=True)

with open(model_path, "wb") as f:
    pickle.dump(clf, f)

print(f"Modelo guardado en: {model_path}")
