# capture_letter.py
import cv2, os, time, csv, argparse
import mediapipe as mp
import numpy as np
from pathlib import Path

# ---------------------------
# CONFIG RUTA ABSOLUTA!!!!!!
# ---------------------------
MODEL_PATH = r"C:\Users\Matías\Desktop\Proyecto AC\src\data_collection\hand_landmarker\float16\hand_landmarker.task"
# ---------------------------

def load_detector(model_path):
    with open(model_path, "rb") as f:
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
    return detector

def mp_image_from_frame(frame):
    # frame en BGR (OpenCV) -> convertir a RGB y crear mp.Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

def ensure_paths(letter):
    base = Path("data/raw")
    imgs_dir = base / letter
    imgs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = base / f"{letter}.csv"
    # si no existe, crear CSV con cabecera
    if not csv_path.exists():
        cols = []
        for i in range(21):
            cols += [f"x{i}", f"y{i}", f"z{i}"]
        cols.append("label")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(cols)
    return imgs_dir, csv_path

def append_row(csv_path, row):
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)
        
def normalize_landmarks(lm_list):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in lm_list])
    wrist = coords[0]
    coords -= wrist
    max_val = np.max(np.abs(coords))
    if max_val > 0:
        coords /= max_val
    return coords.flatten()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--letter", required=True, help="Letra a capturar (A-Z)")
    parser.add_argument("--total", type=int, default=200, help="Total de capturas deseadas para la letra")
    parser.add_argument("--batch", type=int, default=50, help="Cuántas capturas por cada pulsación de 'c'")
    parser.add_argument("--camera", type=int, default=0, help="ID de la cámara (por defecto 0)")
    args = parser.parse_args()

    letter = args.letter.upper()
    total_needed = max(1, args.total)
    batch_size = max(1, args.batch)

    imgs_dir, csv_path = ensure_paths(letter)
    detector = load_detector(MODEL_PATH)

    cap = cv2.VideoCapture(args.camera)
    time.sleep(0.5)

    print(f"\n== Captura para letra '{letter}' ==")
    print(f"Objetivo: {total_needed} imágenes  —  Lote por 'c': {batch_size}")
    print("Presiona C para capturar un lote, ESC para salir.\n")

    saved = len(list(imgs_dir.glob("*.jpg")))  # si ya hay imágenes
    print(f"Ya existen {saved} imágenes en {imgs_dir}")

    window_name = "Capture Letter - Press C"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while saved < total_needed:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.putText(frame, f"Letra:{letter} Guardadas:{saved}/{total_needed}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1)

        # cerrar con ESC o cerrar ventana
        if key == 27:
            print("\nSalida solicitada por usuario (ESC).")
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("\nVentana cerrada. Saliendo.")
            break

        # Capturar lote
        if key == ord('c'):
            remaining = total_needed - saved
            this_batch = min(batch_size, remaining)
            print(f"\nCapturando lote: {this_batch} imágenes...")
            for i in range(this_batch):
                ret, frame2 = cap.read()
                if not ret:
                    continue
                # guardar imagen
                saved += 1
                fname = imgs_dir / f"{letter}_{saved:04d}.jpg"
                cv2.imwrite(str(fname), frame2)

                # extraer landmarks y guardar fila en CSV
                mp_img = mp_image_from_frame(frame2)
                try:
                    res = detector.detect(mp_img)
                except Exception as e:
                    print("Error en detect:", e)
                    # guardar fila vacía si falla
                    row = ["" for _ in range(21*3)] + [letter]
                    append_row(csv_path, row)
                    continue

                if res.hand_landmarks and len(res.hand_landmarks) > 0:
                    lm = res.hand_landmarks[0]
                    row = normalize_landmarks(lm)  # <--- Normalizado
                    row = row.tolist()
                    row.append(letter)
                else:
                    row = [0.0 for _ in range(21*3)] + [letter]
                append_row(csv_path, row)
                print(f"{fname.name}  (landmarks: {len(res.hand_landmarks) if 'res' in locals() else 0})")
                # pequeña pausa para variación natural
                time.sleep(0.04)

            print(f"→ Total guardadas: {saved}/{total_needed}\n")

    cap.release()
    cv2.destroyAllWindows()
    print("\nProceso finalizado.")

if __name__ == "__main__":
    main()
