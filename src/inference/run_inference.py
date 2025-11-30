# src/inference/run_inference.py
import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
from collections import deque

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

MODEL_PATH = "models/letters.pkl"
ENC_PATH = "models/label_encoder.pkl"

def landmarks_to_array(lm):
    return np.array([coord for point in lm for coord in (point.x, point.y, point.z)], dtype=np.float32)

def count_extended_fingers(lm):
    tips_ids = [4,8,12,16,20]
    pip_ids = [2,6,10,14,18]
    count = 0
    for t, p in zip(tips_ids, pip_ids):
        if t == 4:
            if lm[t].x < lm[p].x:
                count += 1
        else:
            if lm[t].y < lm[p].y:
                count += 1
    return count

try:
    clf = joblib.load(MODEL_PATH)
    le = joblib.load(ENC_PATH)
    MODEL_LOADED = True
except Exception:
    clf = None
    le = None
    MODEL_LOADED = False

cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)

writing = False
buffer = []
stable_queue = deque(maxlen=10)
last_action_time = 0
debounce_seconds = 0.6

def search_and_play(query):
    print("Search and Play:", query)
    try:
        import spotipy
        from spotipy.oauth2 import SpotifyOAuth
        sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope="user-modify-playback-state user-read-playback-state"))
        res = sp.search(q=query, type="track", limit=1)
        items = res.get("tracks", {}).get("items", [])
        if items:
            track = items[0]
            uri = track["uri"]
            devices = sp.devices().get("devices", [])
            if devices:
                device_id = devices[0]["id"]
                sp.start_playback(device_id=device_id, uris=[uri])
            else:
                print("No active device found. Open Spotify on a device to play.")
        else:
            print("No results for query:", query)
    except Exception as e:
        print("Spotify integration not available or failed:", e)
        print("Would play:", query)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(img_rgb)
    status_text = "Idle"
    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0].landmark
        mp_drawing.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
        fingers = count_extended_fingers(lm)
        status_text = f"Fingers: {fingers}"
        now = time.time()
        if fingers == 5 and now - last_action_time > debounce_seconds:
            writing = True
            buffer = []
            stable_queue.clear()
            last_action_time = now
            status_text = "Mode: Writing (started)"
        elif fingers == 0 and writing and now - last_action_time > debounce_seconds:
            writing = False
            last_action_time = now
            word = "".join(buffer)
            if word:
                search_and_play(word)
            buffer = []
            stable_queue.clear()
            status_text = f"Confirmed: {word}"
        elif fingers == 1 and now - last_action_time > debounce_seconds:
            last_action_time = now
            status_text = "Toggle Play/Pause (stub)"
            print("Play/Pause toggled (implement Spotify control later)")
        elif fingers == 2 and now - last_action_time > debounce_seconds:
            last_action_time = now
            status_text = "Next track (stub)"
            print("Next track (implement Spotify control later)")
        elif fingers == 3 and now - last_action_time > debounce_seconds:
            last_action_time = now
            status_text = "Previous track (stub)"
            print("Previous track (implement Spotify control later)")
        if writing and MODEL_LOADED:
            arr = landmarks_to_array(lm)
            pred_proba = clf.predict_proba([arr])[0]
            pred_idx = np.argmax(pred_proba)
            pred_label = le.inverse_transform([pred_idx])[0]
            stable_queue.append(pred_label)
            if len(stable_queue) == stable_queue.maxlen and len(set(stable_queue)) == 1:
                buffer.append(pred_label)
                stable_queue.clear()
    cv2.putText(frame, status_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(frame, "Buffer: " + "".join(buffer), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
    cv2.imshow("Inference", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
