import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque
import time

# ===============================
# CONFIG
# ===============================
MODEL_PATH_TASK = r"C:\Users\Matías\Desktop\Proyecto AC\src\data_collection\hand_landmarker\float16\hand_landmarker.task"
MODEL_PATH_KNN = r"C:\Users\Matías\Desktop\Proyecto AC\src\training\swipe_model_knn.pkl"
SEQUENCE_LENGTH = 30
GESTURES = ["swipe_left", "swipe_right", "swipe_up", "swipe_down"]
COOLDOWN = 1.0  # segundos entre detecciones

# ===============================
# LOAD KNN MODEL
# ===============================
clf = joblib.load(MODEL_PATH_KNN)
print("✔ Modelo KNN cargado")

# ===============================
# SETUP MEDIA PIPE HANDLANDMARKER
# ===============================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

with open(MODEL_PATH_TASK, "rb") as f:
    task_data = f.read()

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_buffer=task_data),
    num_hands=1,
    running_mode=VisionRunningMode.IMAGE
)

detector = HandLandmarker.create_from_options(options)

# ===============================
# UTILS
# ===============================
def mp_image_from_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

def normalize_landmarks(lm_list):
    coords = np.array([[lm.x, lm.y] for lm in lm_list])
    wrist = coords[0]
    coords -= wrist  # centrar en la muñeca
    max_val = np.max(np.abs(coords))
    if max_val > 0:
        coords /= max_val  # escalar
    return coords.flatten()

def normalize_sequence(seq):
    seq = np.array(seq)
    if len(seq) != SEQUENCE_LENGTH:
        seq = np.array([seq[int(i*len(seq)/SEQUENCE_LENGTH)] for i in range(SEQUENCE_LENGTH)])
    return seq

# ===============================
# REAL-TIME INFERENCE
# ===============================
cap = cv2.VideoCapture(0)
sequence = deque(maxlen=SEQUENCE_LENGTH)
buffer_predictions = deque(maxlen=5)
last_detection_time = 0

print("Presiona ESC para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    mp_img = mp_image_from_frame(frame)
    result = detector.detect(mp_img)

    if result.hand_landmarks:
        lm = result.hand_landmarks[0]
        vec = normalize_landmarks(lm)
        sequence.append(vec)

        if len(sequence) == SEQUENCE_LENGTH:
            seq_norm = normalize_sequence(list(sequence)).flatten().reshape(1, -1)
            pred = clf.predict(seq_norm)[0]
            buffer_predictions.append(pred)

            if len(buffer_predictions) == buffer_predictions.maxlen:
                current_time = time.time()
                if current_time - last_detection_time >= COOLDOWN:
                    counts = np.bincount(buffer_predictions)
                    gesture_idx = np.argmax(counts)
                    gesture_name = GESTURES[gesture_idx]
                    cv2.putText(frame, f"Swipe: {gesture_name}", (10, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    last_detection_time = current_time
                buffer_predictions.clear()

    cv2.imshow("Swipe Inference", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
