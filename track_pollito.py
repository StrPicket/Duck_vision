import cv2
import math
import numpy as np
import os

VIDEO       = "video_original.mp4"
CM_PER_PX   = 0.2353      # escala: cm/píxel
MIN_MOVE_CM = 0.2         # cm mínimo para contar movimiento

def create_csrt_tracker():
    """
    Crea un tracker CSRT sea que OpenCV esté enlazado bajo cv2.legacy
    o directamente en cv2. Si no existe, lanza RuntimeError.
    """
    # OpenCV >= 4.5 con contrib
    if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
        return cv2.legacy.TrackerCSRT_create()
    # OpenCV con tracker integrado directamente
    if hasattr(cv2, 'TrackerCSRT_create'):
        return cv2.TrackerCSRT_create()
    # Si no está disponible, pedir instalar contrib
    raise RuntimeError(
        "TrackerCSRT_create no encontrado. "
        "Instala opencv-contrib-python:\n"
        "  pip3 uninstall opencv-python\n"
        "  pip3 install opencv-contrib-python"
    )

def main():
    print("Directorio de trabajo:", os.getcwd())
    print("Archivos disponibles:", os.listdir())

    cap = cv2.VideoCapture(VIDEO)
    print("cap.isOpened() =", cap.isOpened())
    if not cap.isOpened():
        print("No puedo abrir el video")
        return

    # Leer primer frame
    ret, frame = cap.read()
    print("Primer frame leído:", ret)
    if not ret:
        print("Error leyendo el video")
        return

    # 1) Seleccionar manualmente las 7 ROI
    bboxes = []
    print("→ Dibuja 7 cajas, una por cada pollito: arrastra, suelta y pulsa ENTER")
    while len(bboxes) < 7:
        roi = cv2.selectROI(f"Pollito #{len(bboxes)}", frame, showCrosshair=True)
        cv2.destroyWindow(f"Pollito #{len(bboxes)}")
        x, y, w, h = roi
        if w == 0 or h == 0:
            print("ROI inválida, inténtalo de nuevo.")
            continue
        bboxes.append((x, y, w, h))

    # 2) Crear un CSRT individual para cada pollito
    trackers = []
    for box in bboxes:
        trk = create_csrt_tracker()
        trk.init(frame, box)
        trackers.append(trk)

    # 3) Variables de distancia y centros previos
    prev_centers = {}
    distances    = {}
    for i, (x, y, w, h) in enumerate(bboxes):
        prev_centers[i] = (x + w/2, y + h/2)
        distances[i]    = 0.0

    # Colores para cada pollito
    colors = [
        (255,   0,   0),
        (  0, 255,   0),
        (  0,   0, 255),
        (255, 255,   0),
        (255,   0, 255),
        (  0, 255, 255),
        (128, 128,   0),
    ]

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = int(1000 / fps)

    # 4) Bucle de tracking
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for i, trk in enumerate(trackers):
            ok, box = trk.update(frame)
            color = colors[i % len(colors)]
            if ok:
                x, y, w, h = box
                cx = x + w/2
                cy = y + h/2

                # Calcular desplazamiento en cm
                dx  = cx - prev_centers[i][0]
                dy  = cy - prev_centers[i][1]
                dcm = math.hypot(dx, dy) * CM_PER_PX

                # Acumular solo si supera el umbral
                if dcm >= MIN_MOVE_CM:
                    distances[i]    += dcm
                    prev_centers[i] = (cx, cy)

                # Dibujar caja e ID
                cv2.rectangle(frame,
                              (int(x), int(y)),
                              (int(x+w), int(y+h)),
                              color, 2)
                cv2.putText(frame, f"#{i}",
                            (int(x), int(y)-5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)
            else:
                # Si perdió, mensaje
                cv2.putText(frame, f"#{i} LOST",
                            (10, 30 + 20*i),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)

        # Mostrar distancias en pantalla
        h_img = frame.shape[0]
        for i in range(7):
            cv2.putText(frame,
                        f"#{i}: {distances[i]:.1f} cm",
                        (10, h_img - 20*(7-i)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, colors[i % len(colors)], 1)

        cv2.imshow("Tracking 7 Pollitos", frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # 5) Imprimir distancias finales
    print("\n=== Distancias finales ===")
    for i in range(7):
        print(f"Pollito #{i}: {distances[i]:.2f} cm recorridos")

if __name__ == "__main__":
    main()
