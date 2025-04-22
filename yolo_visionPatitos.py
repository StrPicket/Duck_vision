# import cv2
# import numpy as np
# from ultralytics import YOLO

# # ——————————————————————————————————————
# # 1) Configuración y detección de rejilla
# # ——————————————————————————————————————
# def detect_grid(frame, 
#                 canny_min=50, canny_max=150, 
#                 hough_thresh=100, 
#                 min_intersections=4):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5,5), 0)
#     edges = cv2.Canny(blur, canny_min, canny_max)

#     # -- Usar HoughLines con un umbral más bajo --
#     lines = cv2.HoughLines(edges, 1, np.pi/180, hough_thresh)
#     if lines is None:
#         raise RuntimeError(
#             f"No se detectaron líneas (HoughThresh={hough_thresh}). "
#             "Prueba bajando ese parámetro o ajustando Canny."
#         )

#     horizontals = []
#     verticals   = []
#     for rho,theta in lines[:,0]:
#         deg = theta*180/np.pi
#         if abs(deg-90) < 10:
#             verticals.append((rho,theta))
#         elif deg < 10 or deg > 170:
#             horizontals.append((rho,theta))

#     if len(horizontals) < 2 or len(verticals) < 2:
#         raise RuntimeError(
#             f"Insuficientes líneas horizontales ({len(horizontals)}) "
#             f"o verticales ({len(verticals)}). Ajusta umbrales."
#         )

#     # calcular intersecciones
#     pts_list = []
#     for r1,t1 in horizontals:
#         for r2,t2 in verticals:
#             A = np.array([[np.cos(t1), np.sin(t1)],
#                           [np.cos(t2), np.sin(t2)]])
#             b = np.array([r1, r2])
#             x0,y0 = np.linalg.solve(A, b)
#             pts_list.append((x0,y0))

#     # si hay muy pocas intersecciones, abortamos
#     if len(pts_list) < min_intersections:
#         raise RuntimeError(
#             f"Solo se detectaron {len(pts_list)} intersecciones. "
#             "Necesitas al menos 4 para estimar el espaciado."
#         )

#     pts = np.array(pts_list)
#     xs = np.sort(pts[:,0])
#     ys = np.sort(pts[:,1])

#     dx = np.diff(xs)
#     dy = np.diff(ys)
#     # descartamos saltos muy pequeños (ruido) y tomamos mediana
#     spacing_x = np.median(dx[dx > 10])
#     spacing_y = np.median(dy[dy > 10])

#     return pts, spacing_x, spacing_y

# # ——————————————————————————————————————
# # 2) Tracker sencillo de centroides
# # ——————————————————————————————————————
# class SimpleTracker:
#     def __init__(self, max_dist=50):
#         self.next_id = 0
#         self.tracks = {}  # id: {'pt':(x,y), 'pix_dist':float}
#         self.max_dist = max_dist

#     def update(self, detections):
#         assigned = {}
#         # detections = list of (cx,cy)
#         for cx,cy in detections:
#             # encontrar track más cercano
#             best_id, best_d = None, float('inf')
#             for tid,info in self.tracks.items():
#                 x0,y0 = info['pt']
#                 d = np.hypot(cx-x0, cy-y0)
#                 if d<best_d and d<self.max_dist:
#                     best_id, best_d = tid, d
#             if best_id is not None:
#                 # asignar a existing track
#                 self.tracks[best_id]['pt'] = (cx,cy)
#                 self.tracks[best_id]['pix_dist'] += best_d
#                 assigned[best_id] = True
#             else:
#                 # crear nuevo track
#                 tid = self.next_id; self.next_id+=1
#                 self.tracks[tid] = {'pt':(cx,cy), 'pix_dist':0.0}
#                 assigned[tid] = True
#         # opcional: limpiar tracks no asignados (o permitir que envejezcan)
#         return self.tracks

# # ——————————————————————————————————————
# # 3) Main
# # ——————————————————————————————————————
# model = YOLO('yolov8n.pt')
# cap   = cv2.VideoCapture('patitos.mp4')
# if not cap.isOpened():
#     raise RuntimeError("No se pudo abrir el video")

# # capturar un frame para detectar la rejilla
# ret, sample = cap.read()
# if not ret:
#     raise RuntimeError("No hay frames en el video")
# grid_pts, spx, spy = detect_grid(sample)
# tile_spacing = (spx + spy)/2.0

# tracker = SimpleTracker(max_dist=50)

# # volver al inicio
# cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # (opcional) dibujar rejilla detectada
#     for (x,y) in grid_pts:
#         cv2.circle(frame, (int(x),int(y)), 3, (255,0,0), -1)

#     # detección YOLO
#     res = model.predict(frame, imgsz=640, conf=0.3, iou=0.4, augment=True)
#     boxes = res[0].boxes.xyxy.cpu().numpy()  # N×4 array

#     centroids = []
#     for x1,y1,x2,y2 in boxes:
#         cx, cy = int((x1+x2)/2), int((y1+y2)/2)
#         centroids.append((cx,cy))
#         cv2.circle(frame, (cx,cy), 4, (0,255,0), -1)

#     tracks = tracker.update(centroids)

#     # mostrar distancia recorrida por cada pato en tiles
#     for tid,info in tracks.items():
#         px = info['pix_dist']
#         tiles = px / tile_spacing
#         x,y = info['pt']
#         text = f"ID{tid}: {tiles:.2f} tiles"
#         cv2.putText(frame, text, (int(x+10), int(y)), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

#     cv2.imshow('Medición en tiles', frame)
#     if cv2.waitKey(1)&0xFF==ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np
from ultralytics import YOLO

# ——————————————————————————————————————
# 1) Configuración y detección de rejilla
# ——————————————————————————————————————
def detect_grid(frame, 
                canny_min=50, canny_max=150, 
                hough_thresh=100, 
                min_intersections=4):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, canny_min, canny_max)
    lines = cv2.HoughLines(edges, 1, np.pi/180, hough_thresh)
    if lines is None:
        raise RuntimeError(
            f"No se detectaron líneas (HoughThresh={hough_thresh}). "
            "Prueba bajando ese parámetro o ajustando Canny."
        )

    horizontals, verticals = [], []
    for rho,theta in lines[:,0]:
        deg = theta*180/np.pi
        if abs(deg-90) < 10:
            verticals.append((rho,theta))
        elif deg < 10 or deg > 200:
            horizontals.append((rho,theta))

    if len(horizontals) < 2 or len(verticals) < 2:
        raise RuntimeError(
            f"Insuficientes líneas horizontales ({len(horizontals)}) "
            f"o verticales ({len(verticals)}). Ajusta umbrales."
        )

    pts_list = []
    for r1,t1 in horizontals:
        for r2,t2 in verticals:
            A = np.array([[np.cos(t1), np.sin(t1)],
                          [np.cos(t2), np.sin(t2)]])
            b = np.array([r1, r2])
            x0,y0 = np.linalg.solve(A, b)
            pts_list.append((x0,y0))

    if len(pts_list) < min_intersections:
        raise RuntimeError(
            f"Solo se detectaron {len(pts_list)} intersecciones. "
            "Necesitas al menos 4 para estimar el espaciado."
        )

    pts = np.array(pts_list)
    xs, ys = np.sort(pts[:,0]), np.sort(pts[:,1])
    dx, dy = np.diff(xs), np.diff(ys)
    spacing_x = np.median(dx[dx > 10])
    spacing_y = np.median(dy[dy > 10])

    return pts, spacing_x, spacing_y

# ——————————————————————————————————————
# 2) Tracker sencillo de centroides
# ——————————————————————————————————————
class SimpleTracker:
    def __init__(self, max_dist=50):
        self.next_id = 0
        self.tracks = {}  # id: {'pt':(x,y), 'pix_dist':float}
        self.max_dist = max_dist

    def update(self, detections):
        assigned = {}
        for cx,cy in detections:
            best_id, best_d = None, float('inf')
            for tid,info in self.tracks.items():
                x0,y0 = info['pt']
                d = np.hypot(cx-x0, cy-y0)
                if d < best_d and d < self.max_dist:
                    best_id, best_d = tid, d
            if best_id is not None:
                self.tracks[best_id]['pt'] = (cx,cy)
                self.tracks[best_id]['pix_dist'] += best_d
                assigned[best_id] = True
            else:
                tid = self.next_id; self.next_id += 1
                self.tracks[tid] = {'pt':(cx,cy), 'pix_dist':0.0}
                assigned[tid] = True
        return self.tracks

# ——————————————————————————————————————
# 3) Main
# ——————————————————————————————————————
model = YOLO('yolov8n.pt')
cap   = cv2.VideoCapture('patitos.mp4')
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir el video")

# 3.1) Estimar rejilla sobre los primeros 100 frames
spx_list, spy_list = [], []
for i in range(100):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret_f, fframe = cap.read()
    if not ret_f:
        break
    try:
        _, spx_i, spy_i = detect_grid(fframe)
        spx_list.append(spx_i)
        spy_list.append(spy_i)
    except RuntimeError as e:
        print(f"[Grid] frame {i}: {e}")

if not spx_list:
    raise RuntimeError("No se pudo estimar la rejilla en los primeros 100 frames")

# promedio de espaciados
spx = np.mean(spx_list)
spy = np.mean(spy_list)
tile_spacing = (spx + spy) / 2.0
print(f"➡️ Espaciado medio entre baldosas: {tile_spacing:.2f} px")

# Reiniciar al inicio para tracking
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
tracker = SimpleTracker(max_dist=50)
frame_idx = 0
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()
    if not ret or frame_idx >= 100:  # si solo quieres medir los primeros 100  
        break

    # — Mostrar número de frame —
    cv2.putText(frame, f"Frame: {frame_idx}", (10,30), font, 1, (255,255,255), 2)

    # (opcional) dibujar puntos de rejilla detectados
    # for (x,y) in grid_pts: cv2.circle(frame, (int(x),int(y)), 3, (255,0,0), -1)

    # — Detección YOLO y centroides —
    res = model.predict(frame, imgsz=640, conf=0.3, iou=0.4, augment=True)
    boxes = res[0].boxes.xyxy.cpu().numpy()
    centroids = []
    for x1,y1,x2,y2 in boxes:
        cx, cy = int((x1+x2)/2), int((y1+y2)/2)
        centroids.append((cx,cy))
        cv2.circle(frame, (cx,cy), 4, (0,255,0), -1)

    # — Actualizar y mostrar distancia en tiles —
    tracks = tracker.update(centroids)
    for tid,info in tracks.items():
        px = info['pix_dist']
        tiles = px / tile_spacing
        x,y = info['pt']
        text = f"ID{tid}: {tiles:.2f} tiles"
        cv2.putText(frame, text, (x+10,y), font, 0.5, (0,255,255), 1)

    cv2.imshow('Medición en tiles', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
