import cv2
import math

# Parámetros globales para guardar los puntos clicados
pts = []

def click_event(event, x, y, flags, param):
    global pts
    if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
        pts.append((x, y))
        cv2.circle(frame_copy, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Frame para calibrar", frame_copy)

# 1. Abrir video y leer el primer frame
cap = cv2.VideoCapture("video_original.mp4")
ret, frame = cap.read()
cap.release()
if not ret:
    print("Error al leer primer frame.")
    exit()

# Hacemos una copia para pintar los clicks
frame_copy = frame.copy()
cv2.imshow("Frame para calibrar", frame_copy)
cv2.setMouseCallback("Frame para calibrar", click_event)

print("Haz click en las DOS ESQUINAS de un mismo lado de un tile.")
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(pts) != 2:
    print("Necesitas clicar exactamente dos puntos.")
    exit()

# 2. Calcular distancia en píxeles
(x1, y1), (x2, y2) = pts
dist_px = math.hypot(x2 - x1, y2 - y1)
print(f"Distancia medida en píxeles: {dist_px:.2f} px")

# 3. Pedir medida real en cm
real_cm = float(input("Introduce la longitud real de ese lado de tile (en cm): "))

# 4. Calcular escala
cm_per_px = real_cm / dist_px
print(f"Escala: {cm_per_px:.4f} cm/px  (o {1/cm_per_px:.2f} px/cm)")
