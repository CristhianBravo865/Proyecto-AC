# search_mode.py
import cv2
import mediapipe as mp
import pickle
import numpy as np
import time

#RUTAS ABSOLUTAS!!!!!!!!!!
MODEL_PATH = r"C:\Users\Matías\Desktop\Proyecto AC\src\data_collection\hand_landmarker\float16\hand_landmarker.task"
TRAINED_MODEL_PATH = r"C:\Users\Matías\Desktop\Proyecto AC\src\training\trained_model.pkl"

# ---------------------------
# Cargar modelo entrenado
# ---------------------------
with open(TRAINED_MODEL_PATH, "rb") as f:
    clf = pickle.load(f)

# ---------------------------
# Configurar MediaPipe
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

# ---------------------------
# Utils
# ---------------------------
def mp_image_from_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

def normalize_landmarks(lm_list):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in lm_list])

    wrist = coords[0]
    coords -= wrist

    max_val = np.max(np.abs(coords))
    if max_val > 0:
        coords /= max_val

    return coords.flatten()

# ---------------------------
# Variables de búsqueda
# ---------------------------
search_mode = False
search_buffer = ""
last_letter = None
letter_start_time = None
HOLD_TIME = 1.3

print("Presiona ESC para salir.")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    mp_img = mp_image_from_frame(frame)
    res = detector.detect(mp_img)

    prediction = "-"
    if res.hand_landmarks:
        vec = normalize_landmarks(res.hand_landmarks[0])
        vec = vec.reshape(1, -1)
        prediction = clf.predict(vec)[0]

    # ===============================
    #        DETECTAR SEARCH
    # ===============================
    if prediction == "SEARCH":
        if not search_mode:
            search_mode = True
            search_buffer = ""
            print("🔍 MODO BÚSQUEDA ACTIVADO")
            time.sleep(1)
        else:
            print(f"✅ PALABRA FINAL: {search_buffer}")
            search_mode = False
            time.sleep(1)

    # ===============================
    #      CAPTURA DE LETRAS
    # ===============================
    if search_mode and prediction not in ["SEARCH", "-"]:
        if prediction != last_letter:
            last_letter = prediction
            letter_start_time = time.time()
        else:
            if time.time() - letter_start_time >= HOLD_TIME:
                search_buffer += prediction
                print("Letra añadida:", prediction)
                last_letter = None
                letter_start_time = None

    # ===============================
    # UI
    # ===============================
    color = (0, 255, 0) if search_mode else (0, 200, 255)
    cv2.putText(frame, f"Prediction: {prediction}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    if search_mode:
        cv2.putText(frame, f"Buffer: {search_buffer}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)

    cv2.imshow("Search Mode", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
