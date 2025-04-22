import cv2
import numpy as np
import matplotlib.pyplot as plt

VIDEO_PATH = 'patitos.mp4'
AREA_THRESHOLD = 500
HOUGH_THRESH = 150
GRID_SQUARE_CM = 10.0

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error al abrir el video.")
    exit()

bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=100, varThreshold=50, detectShadows=False)

pixel_per_cm = None
centroids = []
prev_centroid = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Escala
    if pixel_per_cm is None:
        lines = cv2.HoughLines(edges, 1, np.pi / 180, HOUGH_THRESH)
        if lines is not None:
            rhos = [l[0][0] for l in lines if abs(np.sin(l[0][1])) > 0.9]
            rhos_uniq = sorted(set(int(round(r)) for r in rhos))
            if len(rhos_uniq) >= 2:
                diffs = np.diff(rhos_uniq)
                median_pix = np.median(diffs)
                pixel_per_cm = median_pix / GRID_SQUARE_CM
                print(f"Escala: {pixel_per_cm:.2f} pixeles/cm")

    # Filtro de movimiento + color oscuro
    fg_mask = bg_subtractor.apply(frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 70])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    combined = cv2.bitwise_and(fg_mask, mask_black)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidatos = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < AREA_THRESHOLD:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w//2, y + h//2
        candidatos.append({'cnt': cnt, 'centroid': (cx, cy)})

    elegido = None
    if candidatos:
        if prev_centroid is None:
            elegido = candidatos[0]
        else:
            elegido = min(candidatos, key=lambda c: np.hypot(
                c['centroid'][0] - prev_centroid[0],
                c['centroid'][1] - prev_centroid[1]
            ))

        cx, cy = elegido['centroid']
        centroids.append((cx, cy))
        prev_centroid = (cx, cy)

        x, y, w, h = cv2.boundingRect(elegido['cnt'])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(frame, "Patito", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow('Seguimiento Patito Negro', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Gráfica
if pixel_per_cm and len(centroids) >= 2:
    traj_cm = [(x/pixel_per_cm, y/pixel_per_cm) for x, y in centroids]
    xs, ys = zip(*traj_cm)

    plt.figure(figsize=(6,6))
    plt.plot(xs, ys, '-o', label='Trayectoria')
    for i in range(len(xs)-1):
        dx, dy = xs[i+1]-xs[i], ys[i+1]-ys[i]
        plt.arrow(xs[i], ys[i], dx, dy,
                  head_width=0.5, length_includes_head=True)
    plt.gca().invert_yaxis()
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')
    plt.title('Trayectoria del Patito Negro')
    plt.grid(True)
    plt.legend()
    plt.show()
else:
    print("No se pudo graficar trayectoria.")
