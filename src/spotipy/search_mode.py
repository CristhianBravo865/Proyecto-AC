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
# ESTADOS
# ===============================
STATE_MAIN = 0
STATE_SEARCH_MODE_SELECT = 1
STATE_SEARCH_PLAYLIST_NAME = 2
STATE_SEARCH_TRACK_IN_PLAYLIST = 3
STATE_SEARCH_GLOBAL = 4

state = STATE_MAIN

# ===============================
# RUTAS
# ===============================
MODEL_PATH = r"C:\Users\Matías\Desktop\Proyecto AC\src\data_collection\hand_landmarker\float16\hand_landmarker.task"
TRAINED_MODEL_PATH = r"C:\Users\Matías\Desktop\Proyecto AC\src\training\trained_model.pkl"

# ===============================
# MODELO
# ===============================
with open(TRAINED_MODEL_PATH, "rb") as f:
    clf = pickle.load(f)

# ===============================
# MEDIAPIPE
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
    coords -= coords[0]
    max_val = np.max(np.abs(coords))
    if max_val > 0:
        coords /= max_val
    return coords.flatten()

def detect_motion_gesture(buffer):
    start, end = buffer[0], buffer[-1]
    dx, dy = end[0] - start[0], end[1] - start[1]

    if max(abs(dx), abs(dy)) < MIN_SWIPE_DIST:
        return None

    if abs(dx) > abs(dy) * DIRECTION_RATIO:
        return "NEXT" if dx > 0 else "PREV"

    if abs(dy) > abs(dx) * DIRECTION_RATIO:
        return "PLAY" if dy < 0 else "PAUSE"

    return None

# ===============================
# SEARCH CONFIRM (HOLD)
# ===============================
SEARCH_HOLD_TIME = 2.0
search_hold_start = None

def search_confirmed(prediction):
    global search_hold_start
    if prediction == "SEARCH":
        if search_hold_start is None:
            search_hold_start = time.time()
        elif time.time() - search_hold_start >= SEARCH_HOLD_TIME:
            search_hold_start = None
            return True
    else:
        search_hold_start = None
    return False

# ===============================
# SPOTIFY FUNCIONES
# ===============================
def search_global_track(query):
    sp = get_spotify_client()
    results = sp.search(q=query, type="track", limit=1)
    items = results["tracks"]["items"]
    return items[0] if items else None

def find_user_playlist(name):
    sp = get_spotify_client()
    playlists = sp.current_user_playlists(limit=50)["items"]
    name = name.lower()

    for p in playlists:
        if name in p["name"].lower():
            return p
    return None

def find_track_in_playlist(playlist_uri, track_name):
    sp = get_spotify_client()
    playlist_id = playlist_uri.split(":")[-1]

    track_name = track_name.lower()
    offset = 0
    limit = 100

    while True:
        results = sp.playlist_items(
            playlist_id,
            offset=offset,
            limit=limit,
            additional_types=["track"]
        )

        for item in results["items"]:
            track = item.get("track")
            if track and track_name in track["name"].lower():
                return track

        if results["next"] is None:
            break

        offset += limit

    return None

def play_playlist_from_track(playlist_uri, track_uri):
    sp = get_spotify_client()
    try:
        sp.start_playback(
            context_uri=playlist_uri,
            offset={"uri": track_uri}
        )
    except Exception:
        sp.start_playback(uris=[track_uri])

# ===============================
# VARIABLES TEXTO
# ===============================
buffer_text = ""
last_letter = None
letter_start_time = None
HOLD_TIME = 2.0

selected_playlist = None

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
    lm = None

    if res.hand_landmarks:
        lm = res.hand_landmarks[0]
        vec = normalize_landmarks(lm).reshape(1, -1)
        prediction = clf.predict(vec)[0]

    if state == STATE_MAIN:
        if search_confirmed(prediction):
            state = STATE_SEARCH_MODE_SELECT
            buffer_text = ""
            last_letter = None
            print("MODO BUSQUEDA -> P (Playlist) / B (Global)")
            time.sleep(1)

        if lm is not None:
            center = np.mean([[p.x, p.y, p.z] for p in lm], axis=0)
            motion_buffer.append(center)

            if len(motion_buffer) == motion_buffer.maxlen:
                if time.time() - last_gesture_time > GESTURE_COOLDOWN:
                    g = detect_motion_gesture(motion_buffer)
                    if g:
                        sp = get_spotify_client()
                        if g == "NEXT": sp.next_track()
                        elif g == "PREV": sp.previous_track()
                        elif g == "PLAY": sp.start_playback()
                        elif g == "PAUSE": sp.pause_playback()
                        last_gesture_time = time.time()
                        motion_buffer.clear()
        else:
            motion_buffer.clear()

    elif state == STATE_SEARCH_MODE_SELECT:
        if prediction in ["P", "B"]:
            if prediction != last_letter:
                last_letter = prediction
                letter_start_time = time.time()
            elif time.time() - letter_start_time >= HOLD_TIME:
                state = STATE_SEARCH_PLAYLIST_NAME if prediction == "P" else STATE_SEARCH_GLOBAL
                buffer_text = ""
                last_letter = None
                letter_start_time = None
                time.sleep(1)

    elif state == STATE_SEARCH_GLOBAL:
        if search_confirmed(prediction):
            track = search_global_track(buffer_text)
            if track:
                get_spotify_client().start_playback(uris=[track["uri"]])
            state = STATE_MAIN

        elif prediction not in ["-", "SEARCH"]:
            if prediction != last_letter:
                last_letter = prediction
                letter_start_time = time.time()
            elif time.time() - letter_start_time >= HOLD_TIME:
                buffer_text += prediction
                last_letter = None
                letter_start_time = None

    elif state == STATE_SEARCH_PLAYLIST_NAME:
        if search_confirmed(prediction):
            playlist = find_user_playlist(buffer_text)
            if playlist:
                selected_playlist = playlist
                print("Playlist:", playlist["name"])
                state = STATE_SEARCH_TRACK_IN_PLAYLIST
            else:
                state = STATE_MAIN

            buffer_text = ""
            last_letter = None
            letter_start_time = None
            time.sleep(1)

        elif prediction not in ["-", "SEARCH"]:
            if prediction != last_letter:
                last_letter = prediction
                letter_start_time = time.time()
            elif time.time() - letter_start_time >= HOLD_TIME:
                buffer_text += prediction
                last_letter = None
                letter_start_time = None

    elif state == STATE_SEARCH_TRACK_IN_PLAYLIST:
        if search_confirmed(prediction):
            if selected_playlist:
                track = find_track_in_playlist(
                    selected_playlist["uri"],
                    buffer_text
                )
                if track:
                    play_playlist_from_track(
                        selected_playlist["uri"],
                        track["uri"]
                    )
                else:
                    print("No se encontró la canción en la playlist")

            selected_playlist = None
            buffer_text = ""
            last_letter = None
            letter_start_time = None
            state = STATE_MAIN
            time.sleep(1)

        elif prediction not in ["-", "SEARCH"]:
            if prediction != last_letter:
                last_letter = prediction
                letter_start_time = time.time()
            elif time.time() - letter_start_time >= HOLD_TIME:
                buffer_text += prediction
                last_letter = None
                letter_start_time = None

    cv2.putText(frame, f"State: {state}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Prediction: {prediction}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Buffer: {buffer_text}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if selected_playlist:
        cv2.putText(frame, f"Playlist: {selected_playlist['name']}", (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Hand Spotify Control", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
