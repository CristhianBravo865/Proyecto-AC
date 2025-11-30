# infer_letter.py
import cv2
import mediapipe as mp
import pickle
import numpy as np
from pathlib import Path

# ---------------------------
# CONFIG
# ---------------------------
MODEL_PATH = r"C:\Users\Matías\Desktop\Proyecto AC\src\data_collection\hand_landmarker\float16\hand_landmarker.task"
TRAINED_MODEL_PATH = r"C:\Users\Matías\Desktop\Proyecto AC\src\training\trained_model.pkl"
NUM_LANDMARKS = 21

# ---------------------------
# Cargar modelo entrenado
# ---------------------------
with open(TRAINED_MODEL_PATH, "rb") as f:
    clf = pickle.load(f)

# ---------------------------
# Configurar MediaPipe HandLandmarker
# ---------------------------
with open(MODEL_PATH, "rb") as f:
    model_data = f.read()

BaseOptions = mp.tasks.BaseOptions
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

base_options = BaseOptions(model_asset_buffer=model_data)
options = HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=VisionRunningMode.IMAGE
)

detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

def mp_image_from_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

def landmarks_to_vector(lm_list):
    """Convertir landmarks de MediaPipe a vector 1D para el modelo"""
    vec = []
    for lm in lm_list:
        vec.extend([lm.x, lm.y, lm.z])
    return vec

# ---------------------------
# Captura en tiempo real
# ---------------------------
cap = cv2.VideoCapture(0)
cv2.namedWindow("Inferencia Letra", cv2.WINDOW_NORMAL)

print("Presiona ESC para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    mp_img = mp_image_from_frame(frame)
    try:
        res = detector.detect(mp_img)
    except Exception as e:
        print("Error en detección:", e)
        continue

    pred_letter = "-"
    if res.hand_landmarks and len(res.hand_landmarks) > 0:
        vec = landmarks_to_vector(res.hand_landmarks[0])
        vec_np = np.array(vec).reshape(1, -1)
        pred_letter = clf.predict(vec_np)[0]

        # Dibujar landmarks
        for lm in res.hand_landmarks[0]:
            h, w, _ = frame.shape
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # Mostrar predicción en pantalla
    cv2.putText(frame, f"Letra: {pred_letter}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

    cv2.imshow("Inferencia Letra", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
