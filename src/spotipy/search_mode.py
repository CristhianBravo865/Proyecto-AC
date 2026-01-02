import cv2
import mediapipe as mp
import pickle
import numpy as np
import time
import pyttsx3 
import threading
import queue

from spotify_auth import get_spotify_client
from difflib import SequenceMatcher

# ===============================
# CONFIG GESTOS ESTÁTICOS
# ===============================
GESTURE_COOLDOWN = 1.2
HOLD_TIME = 1.5  # tiempo que debe mantenerse el gesto
last_gesture_time = 0
last_gesture = None
gesture_start_time = None
VALID_LETTERS = set(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))

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
def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

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

# ===============================
# GESTOS ESTÁTICOS
# ===============================
def detect_static_gesture(prediction):
    if prediction == "APUNTAR_DERECHA":
        return "NEXT"
    elif prediction == "APUNTAR_IZQUIERDA":
        return "PREV"
    elif prediction == "PAUSA_PLAY":
        return "PLAY/PAUSE"
    elif prediction == "APUNTAR_ARRIBA":
        return "VOLUME_UP"
    elif prediction == "APUNTAR_ABAJO":
        return "VOLUME_DOWN"
    return None

# ===============================
# SPOTIFY FUNCIONES
# ===============================
def find_best_track_in_playlist(playlist_uri, query):
    sp = get_spotify_client()
    playlist_id = playlist_uri.split(":")[-1]
    query = query.lower()

    offset = 0
    limit = 100

    best_track = None
    best_score = 0

    while True:
        results = sp.playlist_items(
            playlist_id,
            offset=offset,
            limit=limit,
            additional_types=["track"]
        )

        for item in results["items"]:
            track = item.get("track")
            if not track or not track.get("name"):
                continue

            name = track["name"].lower()

            artists_list = track.get("artists", [])
            artists = " ".join(
                a["name"].lower()
                for a in artists_list
                if a.get("name")
            )

            title_words = name.split()
            first_word = title_words[0] if title_words else ""

            score_name = similarity(query, name)
            score_artist = similarity(query, artists)
            score_first_word = similarity(query, first_word)

            bonus = 0
            if first_word.startswith(query):
                bonus += 0.15
            elif query in first_word:
                bonus += 0.08

            score = max(score_name, score_artist, score_first_word) + bonus

            if score > best_score:
                best_score = score
                best_track = track

        if results["next"] is None:
            break

        offset += limit

    return best_track, best_score

def get_current_volume(sp):
    playback = sp.current_playback()
    if playback and "device" in playback and "volume_percent" in playback["device"]:
        return playback["device"]["volume_percent"]
    return 50

sp = get_spotify_client()
current_volume = get_current_volume(sp)

def control_spotify(gesture):
    global current_volume
    sp = get_spotify_client()
    if gesture == "NEXT":
        sp.next_track()
    elif gesture == "PREV":
        sp.previous_track()
    elif gesture == "PLAY/PAUSE":
        sp_playback = sp.current_playback()
        if sp_playback and sp_playback["is_playing"]:
            sp.pause_playback()
        else:
            sp.start_playback()
    elif gesture == "VOLUME_UP":
        current_volume = min(current_volume + 10, 100)
        sp.volume(current_volume)
    elif gesture == "VOLUME_DOWN":
        current_volume = max(current_volume - 10, 0)
        sp.volume(current_volume)

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
# SPOTIFY BÚSQUEDA
# ===============================
def find_user_playlist(name):
    sp = get_spotify_client()
    playlists = sp.current_user_playlists(limit=50)["items"]
    name = name.lower()
    for p in playlists:
        if name in p["name"].lower():
            return p
    return None

def play_playlist_from_track(playlist_uri, track_uri):
    sp = get_spotify_client()
    try:
        sp.start_playback(context_uri=playlist_uri, offset={"uri": track_uri})
    except Exception:
        sp.start_playback(uris=[track_uri])

# ===============================
# VARIABLES TEXTO
# ===============================
buffer_text = ""
last_letter = None
letter_start_time = None
selected_playlist = None

# ===============================
# VOZ NO BLOQUEANTE
# ===============================
voice_queue = queue.Queue()

def voice_worker():
    while True:
        text = voice_queue.get()
        if text is None:
            break
        
        try:
            # Inicializamos el motor DENTRO del bucle para cada frase
            engine = pyttsx3.init()
            engine.setProperty('rate', 125) # Velocidad
            engine.setProperty('volume', 1)
            
            engine.say(text)
            engine.runAndWait()
            
            # Forzamos la parada y eliminación del objeto para liberar el COM
            engine.stop()
            del engine 
        except Exception as e:
            print(f"Error en el motor de voz: {e}")


voice_thread = threading.Thread(target=voice_worker, daemon=True)
voice_thread.start()

def speak(text):
    voice_queue.put(text)

print("Presiona ESC para salir")
cap = cv2.VideoCapture(0)

# ===============================
# BACKSPACE GESTUAL
# ===============================
def handle_backspace():
    global buffer_text, state, selected_playlist
    if buffer_text:
        buffer_text = buffer_text[:-1]
        print(" Letra eliminada. Buffer:", buffer_text)
        speak("Letra eliminada. Buffer actualizado.")
    else:
        # Volver al estado anterior
        if state == STATE_SEARCH_PLAYLIST_NAME or state == STATE_SEARCH_GLOBAL:
            state = STATE_SEARCH_MODE_SELECT
            print("↩ Volviendo a selección de modo búsqueda")
            speak("Volviendo a selección de modo búsqueda.")
        elif state == STATE_SEARCH_TRACK_IN_PLAYLIST:
            state = STATE_SEARCH_PLAYLIST_NAME
            selected_playlist = None
            print("↩ Volviendo a búsqueda de playlist")
            speak("Volviendo a búsqueda de playlist.")

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

    # ===============================
    # ESTADO PRINCIPAL (GESTOS ESTÁTICOS)
    # ===============================
    if state == STATE_MAIN:
        gesture = detect_static_gesture(prediction)

        if gesture:
            if last_gesture != gesture:
                last_gesture = gesture
                gesture_start_time = time.time()
            elif time.time() - gesture_start_time >= HOLD_TIME:
                if time.time() - last_gesture_time >= GESTURE_COOLDOWN:
                    control_spotify(gesture)
                    last_gesture_time = time.time()
                    gesture_start_time = None
                    last_gesture = None
        else:
            last_gesture = None
            gesture_start_time = None

        if search_confirmed(prediction):
            state = STATE_SEARCH_MODE_SELECT
            buffer_text = ""
            last_letter = None
            speak("Modo búsqueda activado. Elige P para Playlist o B para búsqueda global.")
            print("MODO BÚSQUEDA -> P (Playlist) / B (Global)")
            time.sleep(1)

    # ===============================
    # ESTADOS DE BÚSQUEDA
    # ===============================
    elif state == STATE_SEARCH_MODE_SELECT:
        if prediction in ["P", "B", "APUNTAR_IZQUIERDA"]:
            if prediction != last_letter:
                last_letter = prediction
                letter_start_time = time.time()
            elif time.time() - letter_start_time >= HOLD_TIME:
                if prediction == "P":
                    state = STATE_SEARCH_PLAYLIST_NAME
                    speak("Modo búsqueda: Playlist. Escribe el nombre de la playlist.")
                    print("Modo búsqueda: Playlist")
                elif prediction == "B":
                    state = STATE_SEARCH_GLOBAL
                    speak("Modo búsqueda: Global. Escribe el nombre de la canción.")
                    print("Modo búsqueda: Global")
                elif prediction == "APUNTAR_IZQUIERDA":
                    state = STATE_MAIN
                    speak("Volviendo al estado principal.")
                    print("↩ Volviendo al estado principal")
                buffer_text = ""
                last_letter = None
                letter_start_time = None
                time.sleep(0.5)

    elif state == STATE_SEARCH_GLOBAL:

        if prediction == "APUNTAR_IZQUIERDA":
            if last_letter != prediction:
                last_letter = prediction
                letter_start_time = time.time()
            elif time.time() - letter_start_time >= HOLD_TIME:
                handle_backspace()
                last_letter = None
                letter_start_time = None

        elif prediction in VALID_LETTERS:
            if prediction != last_letter:
                last_letter = prediction
                letter_start_time = time.time()
            elif time.time() - letter_start_time >= HOLD_TIME:
                buffer_text += prediction
                speak(f"Letra guardada: {prediction}")
                last_letter = None
                letter_start_time = None

        if search_confirmed(prediction) and buffer_text:
            sp = get_spotify_client()
            results = sp.search(q=buffer_text, type="track", limit=10)
            items = results["tracks"]["items"]

            best_match = None
            best_score = 0
            query = buffer_text.lower()

            for t in items:
                track_name = t["name"].lower()
                artist_name = " ".join(a["name"].lower() for a in t["artists"])

                score_name = similarity(query, track_name)
                score_artist = similarity(query, artist_name)

                score = max(score_name, score_artist)

                if score > best_score:
                    best_score = score
                    best_match = t

            if best_match and best_score > 0.1:
                print(f"Reproduciendo: {best_match['name']} - {best_match['artists'][0]['name']}")
                sp.start_playback(uris=[best_match["uri"]])
                speak(f"Reproduciendo: {best_match['name']} de {best_match['artists'][0]['name']}")
            else:
                print("No se encontró ninguna canción relevante.")
                speak("No se encontró ninguna canción relevante.")

            state = STATE_MAIN
            buffer_text = ""
            last_letter = None
            letter_start_time = None

    elif state == STATE_SEARCH_PLAYLIST_NAME:
        if prediction not in ["-", "SEARCH"]:
            if prediction == "APUNTAR_IZQUIERDA":
                if last_letter != prediction:
                    last_letter = prediction
                    letter_start_time = time.time()
                elif time.time() - letter_start_time >= HOLD_TIME:
                    handle_backspace()
                    last_letter = None
                    letter_start_time = None
            else:
                if prediction != last_letter:
                    last_letter = prediction
                    letter_start_time = time.time()
                elif time.time() - letter_start_time >= HOLD_TIME:
                    buffer_text += prediction
                    speak(f"Letra guardada: {prediction}")
                    last_letter = None
                    letter_start_time = None

        if search_confirmed(prediction):
            playlist = find_user_playlist(buffer_text)
            if playlist:
                selected_playlist = playlist
                speak(f"Playlist encontrada: {playlist['name']}")
                print(f"Playlist: {playlist['name']}")
                state = STATE_SEARCH_TRACK_IN_PLAYLIST
            else:
                speak("No se encontró la playlist.")
                state = STATE_MAIN
            buffer_text = ""
            last_letter = None
            letter_start_time = None
            time.sleep(1)

    elif state == STATE_SEARCH_TRACK_IN_PLAYLIST:
        if prediction not in ["-", "SEARCH"]:
            if prediction == "APUNTAR_IZQUIERDA":
                if last_letter != prediction:
                    last_letter = prediction
                    letter_start_time = time.time()
                elif time.time() - letter_start_time >= HOLD_TIME:
                    handle_backspace()
                    last_letter = None
                    letter_start_time = None
            else:
                if prediction != last_letter:
                    last_letter = prediction
                    letter_start_time = time.time()
                elif time.time() - letter_start_time >= HOLD_TIME:
                    buffer_text += prediction
                    speak(f"Letra guardada: {prediction}")
                    last_letter = None
                    letter_start_time = None

        if search_confirmed(prediction):
            if selected_playlist:
                track, score = find_best_track_in_playlist(
                    selected_playlist["uri"],
                    buffer_text
                )

                if track and score > 0.25:
                    play_playlist_from_track(selected_playlist["uri"], track["uri"])
                    speak(f"Reproduciendo {track['name']} de {track['artists'][0]['name']}")
                else:
                    speak("No se encontró una canción relevante en la playlist.")

            selected_playlist = None
            buffer_text = ""
            last_letter = None
            letter_start_time = None
            state = STATE_MAIN
            time.sleep(1)

    # ===============================
    # UI
    # ===============================
    cv2.putText(frame, f"State: {state}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Prediction: {prediction}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Buffer: {buffer_text}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    if selected_playlist:
        cv2.putText(frame, f"Playlist: {selected_playlist['name']}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Hand Spotify Control", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
voice_queue.put(None)

