import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ===============================
# CONFIG
# ===============================
DATA_DIR = "gesture_data"
GESTURES = {"1":"swipe_left","2":"swipe_right","3":"swipe_up","4":"swipe_down"}
SEQUENCE_LENGTH = 30
MODEL_PATH_TASK = r"src/data_collection/hand_landmarker/float16/hand_landmarker.task"

# ===============================
# MEDIA PIPE TASKS
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
    coords = np.array([[lm.x, lm.y] for lm in lm_list])  # solo x,y
    wrist = coords[0]  # landmark 0
    coords -= wrist  # centrar en la muñeca
    max_val = np.max(np.abs(coords))
    if max_val > 0:
        coords /= max_val
    return coords.flatten()

def normalize_sequence(seq):
    seq = np.array(seq)
    if len(seq) != SEQUENCE_LENGTH:
        seq = np.array([seq[int(i*len(seq)/SEQUENCE_LENGTH)] for i in range(SEQUENCE_LENGTH)])
    return seq.flatten()

# ===============================
# PREPARE DIRS
# ===============================
os.makedirs(DATA_DIR, exist_ok=True)
for g in GESTURES.values():
    os.makedirs(os.path.join(DATA_DIR, g), exist_ok=True)

# ===============================
# STATE
# ===============================
current_gesture = None
recording = False
sequence = []

print("Controles: 1-4 seleccionar gesto | r grabar | s guardar | ESC salir")
cap = cv2.VideoCapture(0)

# ===============================
# MAIN LOOP
# ===============================
while True:
    ret, frame = cap.read()
    if not ret: continue

    mp_img = mp_image_from_frame(frame)
    result = detector.detect(mp_img)

    if recording and result.hand_landmarks:
        lm = result.hand_landmarks[0]
        sequence.append(normalize_landmarks(lm))

        if len(sequence) >= SEQUENCE_LENGTH:
            recording = False
            print("✔ Secuencia completa")

    cv2.putText(frame, f"Gesto: {current_gesture}", (10,40), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.putText(frame, f"Grabando: {recording}", (10,80), cv2.FONT_HERSHEY_SIMPLEX,1,(0,200,255),2)
    cv2.putText(frame, f"Frames: {len(sequence)}/{SEQUENCE_LENGTH}", (10,120), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)

    cv2.imshow("Swipe Recorder", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break
    if chr(key) in GESTURES:
        current_gesture = GESTURES[chr(key)]
        sequence = []
        print("▶ Gesto seleccionado:", current_gesture)
    if key == ord("r") and current_gesture:
        sequence = []
        recording = True
        print("● Grabando...")
    if key == ord("s") and sequence and current_gesture:
        seq_norm = normalize_sequence(sequence)
        path = os.path.join(DATA_DIR,current_gesture,f"{int(time.time())}.npy")
        np.save(path, seq_norm)
        sequence = []
        print("💾 Guardado:", path)

cap.release()
cv2.destroyAllWindows()
