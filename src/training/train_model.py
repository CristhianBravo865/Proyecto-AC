import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from pathlib import Path
import glob

# ============================
# Ruta de los CSVs
# ============================
BASE_DIR = Path(__file__).resolve().parent  # src/training
DATA_DIR = BASE_DIR.parent / "data_collection" / "data" / "raw"
data_files = glob.glob(str(DATA_DIR / "*.csv"))

if not data_files:
    raise FileNotFoundError(f"No se encontraron CSV en {DATA_DIR}")

print("Cargando archivos:")
for f in data_files:
    print(" -", f)

# ============================
# Cargar y preparar datos
# ============================
data = pd.concat([pd.read_csv(f) for f in data_files], ignore_index=True)
X = data.drop("label", axis=1)
y = data["label"]

# ============================
# Entrenar modelo
# ============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=300, random_state=42)
clf.fit(X_train, y_train)

# ============================
# Evaluación
# ============================
y_pred = clf.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# ============================
# Guardar modelo
# ============================
MODEL_PATH = BASE_DIR.parent / "training" / "trained_model.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(clf, f)

print(f"Modelo guardado en: {MODEL_PATH}")
