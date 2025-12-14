import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ===============================
# CONFIG
# ===============================
DATA_DIR = "gesture_data"
MODEL_SAVE_PATH = r"C:\Users\Matías\Desktop\Proyecto AC\src\training\swipe_model_knn.pkl"
SEQUENCE_LENGTH = 30
GESTURE_LABELS = {"swipe_left":0, "swipe_right":1, "swipe_up":2, "swipe_down":3}

# ===============================
# CARGAR DATOS
# ===============================
X, y = [], []

for gesture_name, label in GESTURE_LABELS.items():
    gesture_dir = os.path.join(DATA_DIR, gesture_name)
    for file in os.listdir(gesture_dir):
        if file.endswith(".npy"):
            seq = np.load(os.path.join(gesture_dir, file))
            X.append(seq)
            y.append(label)

X = np.array(X)
y = np.array(y)

print(f"✔ Cargadas {len(X)} secuencias")

# ===============================
# DIVIDIR DATOS
# ===============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ===============================
# ENTRENAR KNN
# ===============================
knn = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
knn.fit(X_train, y_train)

# ===============================
# EVALUACIÓN
# ===============================
y_pred = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=GESTURE_LABELS.keys()))

# ===============================
# GUARDAR MODELO
# ===============================
joblib.dump(knn, MODEL_SAVE_PATH)
print("💾 Modelo guardado en:", MODEL_SAVE_PATH)
