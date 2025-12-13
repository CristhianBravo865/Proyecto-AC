import cv2
import mediapipe as mp
import pickle
import numpy as np
import time
from spotify_auth import get_spotify_client
from collections import deque

# ===============================
# CONFIG GESTOS DINÁMICOS
# ===============================
motion_buffer = deque(maxlen=12)
last_gesture_time = 0
GESTURE_COOLDOWN = 1.2

MIN_SWIPE_DIST = 0.14
DIRECTION_RATIO = 2.0

# ===============================
# RUTAS ABSOLUTAS
# ===============================
MODEL_PATH = r"C:\Users\Matías\Desktop\Proyecto AC\src\data_collection\hand_landmarker\float16\hand_landmarker.task"
TRAINED_MODEL_PATH = r"C:\Users\Matías\Desktop\Proyecto AC\src\training\trained_model.pkl"

# ===============================
# CARGAR MODELO ENTRENADO
# ===============================
with open(TRAINED_MODEL_PATH, "rb") as f:
    clf = pickle.load(f)

# ===============================
# MEDIAPIPE HAND LANDMARKER
# ===============================
with open(MODEL_PATH, "rb") as f:
    model_data = f.read()

BaseOptions = mp.tasks.BaseOptions
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_buffer=model_data),
    num_hands=1,
    running_mode=VisionRunningMode.IMAGE
)

detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

# ===============================
# UTILS
# ===============================
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

def is_palm_open(lm):
    tips = [8, 12, 16, 20]
    open_fingers = 0
    for tip in tips:
        if lm[tip].y < lm[tip - 2].y:
            open_fingers += 1
    return open_fingers >= 3

def detect_motion_gesture(buffer):
    start = buffer[0]
    end = buffer[-1]

    dx = end[0] - start[0]
    dy = end[1] - start[1]

    abs_dx = abs(dx)
    abs_dy = abs(dy)

    if max(abs_dx, abs_dy) < MIN_SWIPE_DIST:
        return None

    if abs_dx > abs_dy * DIRECTION_RATIO:
        return "NEXT" if dx > 0 else "PREV"

    if abs_dy > abs_dx * DIRECTION_RATIO:
        return "PLAY" if dy < 0 else "PAUSE"

    return None

# ===============================
# SPOTIFY SEARCH
# ===============================
def spotify_play_from_word(word):
    try:
        sp = get_spotify_client()
        results = sp.search(q=f"track:{word}", type="track", limit=10)
        items = results["tracks"]["items"]

        if not items:
            print("No se encontró:", word)
            return

        def similarity(a, b):
            return sum(1 for x, y in zip(a, b) if x == y) / max(len(a), len(b))

        best_track = max(
            items,
            key=lambda t: similarity(word.lower(), t["name"].lower())
        )

        print("Reproduciendo:", best_track["name"], "-", best_track["artists"][0]["name"])
        sp.start_playback(uris=[best_track["uri"]])

    except Exception as e:
        print("Error Spotify:", e)

# ===============================
# VARIABLES DE ESTADO
# ===============================
search_mode = False
search_buffer = ""
last_letter = None
letter_start_time = None

HOLD_TIME = 2.0
SEARCH_HOLD_TIME = 2.0
search_start_time = None

print("Presiona ESC para salir")

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
        vec = normalize_landmarks(res.hand_landmarks[0]).reshape(1, -1)
        prediction = clf.predict(vec)[0]

    # ==================================================
    # ESTADO 1 -> MODO PRINCIPAL
    # ==================================================
    if not search_mode:

        if prediction == "SEARCH":
            if search_start_time is None:
                search_start_time = time.time()
            elif time.time() - search_start_time >= SEARCH_HOLD_TIME:
                search_mode = True
                search_buffer = ""
                last_letter = None
                letter_start_time = None
                print("MODO BUSQUEDA ACTIVADO")
                time.sleep(1)
        else:
            search_start_time = None

        if res.hand_landmarks:
            lm = res.hand_landmarks[0]

            if is_palm_open(lm):
                coords = np.array([[p.x, p.y, p.z] for p in lm])
                center = coords.mean(axis=0)
                motion_buffer.append(center)

                if len(motion_buffer) == motion_buffer.maxlen:
                    if time.time() - last_gesture_time > GESTURE_COOLDOWN:
                        gesture = detect_motion_gesture(motion_buffer)

                        if gesture:
                            sp = get_spotify_client()

                            if gesture == "NEXT":
                                print("SIGUIENTE")
                                sp.next_track()

                            elif gesture == "PREV":
                                print("ANTERIOR")
                                sp.previous_track()

                            elif gesture == "PLAY":
                                print("PLAY")
                                sp.start_playback()

                            elif gesture == "PAUSE":
                                print("PAUSA")
                                sp.pause_playback()

                            last_gesture_time = time.time()
                            motion_buffer.clear()
            else:
                motion_buffer.clear()

    # ==================================================
    # ESTADO 2 -> MODO BUSQUEDA
    # ==================================================
    else:
        if prediction == "SEARCH":
            if search_start_time is None:
                search_start_time = time.time()
            elif time.time() - search_start_time >= SEARCH_HOLD_TIME:
                print("PALABRA FINAL:", search_buffer)
                spotify_play_from_word(search_buffer)
                search_mode = False
                time.sleep(1)
        else:
            search_start_time = None

        if prediction not in ["SEARCH", "-"]:
            if prediction != last_letter:
                last_letter = prediction
                letter_start_time = time.time()
            elif time.time() - letter_start_time >= HOLD_TIME:
                search_buffer += prediction
                print("Letra:", prediction)
                last_letter = None
                letter_start_time = None

    # ==================================================
    # UI
    # ==================================================
    cv2.putText(frame, f"Prediction: {prediction}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if search_mode:
        cv2.putText(frame, f"Buffer: {search_buffer}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
    else:
        cv2.putText(frame, "MODO PRINCIPAL", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Hand Search Spotify", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
