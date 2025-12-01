# search_mode.py
import cv2
import mediapipe as mp
import pickle
import numpy as np
import time
from spotify_auth import get_spotify_client

# RUTAS ABSOLUTAS !!!!!!!
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
# Spotify Search & Play
# ---------------------------
def spotify_play_from_word(word):
    try:
        sp = get_spotify_client()

        # Buscar más resultados para tener variedad
        results = sp.search(q=word, type="track", limit=10)
        items = results["tracks"]["items"]

        if not items:
            print("No se encontró nada en Spotify para:", word)
            return

        # Función simple de similitud: proporción de coincidencia
        def similarity(a, b):
            a = a.lower()
            b = b.lower()
            matches = sum(1 for x, y in zip(a, b) if x == y)
            return matches / max(len(a), len(b))

        best_track = None
        best_score = -1
        target = word.lower()

        for track in items:
            name = track["name"].lower()

            # Coincidencia exacta → escoger inmediatamente
            if name == target:
                best_track = track
                break

            # Coincidencia aproximada → medir similitud
            score = similarity(target, name)

            if score > best_score:
                best_score = score
                best_track = track

        if best_track:
            print(f"Reproduciendo: {best_track['name']} - {best_track['artists'][0]['name']}")
            sp.start_playback(uris=[best_track["uri"]])
        else:
            print("No se pudo determinar la mejor coincidencia.")

    except Exception as e:
        print("Error al reproducir en Spotify:", e)

# ---------------------------
# Variables de búsqueda
# ---------------------------
search_mode = False
search_buffer = ""
last_letter = None
letter_start_time = None
HOLD_TIME = 2.0

SEARCH_HOLD_TIME = 2.0
search_start_time = None
search_detected_once = False

print("Presiona ESC para salir.")

cap = cv2.VideoCapture(0)

# ===============================
# MAIN LOOP
# ===============================
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
    # DETECTAR SEARCH GESTURE CON HOLD
    # ===============================
    if prediction == "SEARCH":

        if search_start_time is None:
            search_start_time = time.time()

        else:
            if time.time() - search_start_time >= SEARCH_HOLD_TIME:

                if not search_mode:  
                    search_mode = True
                    search_buffer = ""
                    print("MODO BÚSQUEDA ACTIVADO")
                    search_detected_once = True
                    time.sleep(1)

                else:
                    if not search_detected_once:
                        print(f"PALABRA FINAL: {search_buffer}")
                        print("Enviando a Spotify...")
                        spotify_play_from_word(search_buffer)
                        search_mode = False
                        time.sleep(1)
                    search_detected_once = False

    else:
        search_start_time = None

    # ===============================
    # LETRAS EN MODO BÚSQUEDA
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
