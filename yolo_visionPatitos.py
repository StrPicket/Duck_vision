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
import matplotlib.pyplot as plt
from ultralytics import YOLO
import supervision as sv
from typing import List, Tuple

# --- Parámetros ajustables ---
VIDEO_PATH = 'patitos.mp4'
AREA_THRESHOLD = 500      # área mínima para considerar un contorno
HOUGH_THRESH = 150        # umbral de HoughLines
GRID_SQUARE_CM = 10.0     # tamaño real de cada cuadrado de la malla
CONFIDENCE_THRESH = 0.5   # umbral de confianza para detecciones YOLO
IOU_THRESH = 0.5          # umbral IoU para NMS

# --- Clase para seguimiento de objetos ---
class PathTracker:
    def __init__(self):
        self.paths = {}  # diccionario para almacenar trayectorias
        self.disappeared = {}  # contador de frames desaparecidos
        self.next_id = 0  # próximo ID a asignar
        self.max_disappeared = 10  # máximo de frames para mantener un ID desaparecido
    
    def update(self, detections):
        # Si no hay detecciones, incrementar contadores de desaparición
        if len(detections) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    del self.paths[obj_id]
                    del self.disappeared[obj_id]
            return self.paths
        
        # Si es la primera detección, inicializar los objetos
        if len(self.paths) == 0:
            for detection in detections:
                self.paths[self.next_id] = [detection]
                self.disappeared[self.next_id] = 0
                self.next_id += 1
        else:
            # Calcular distancias entre detecciones actuales y trayectorias existentes
            object_ids = list(self.paths.keys())
            previous_centroids = [self.paths[obj_id][-1] for obj_id in object_ids]
            
            # Matriz de distancias entre detecciones actuales y objetos existentes
            distances = np.zeros((len(previous_centroids), len(detections)))
            for i, prev_centroid in enumerate(previous_centroids):
                for j, detection in enumerate(detections):
                    distances[i, j] = np.linalg.norm(
                        np.array(prev_centroid[:2]) - np.array(detection[:2])
                    )
            
            # Asignar detecciones a trayectorias existentes
            rows_idx = list(range(distances.shape[0]))
            cols_idx = list(range(distances.shape[1]))
            
            if distances.size > 0:
                # Mientras haya posibles asignaciones
                while len(rows_idx) > 0 and len(cols_idx) > 0:
                    # Encontrar par con menor distancia
                    if len(rows_idx) > 0 and len(cols_idx) > 0:
                        min_idx = np.argmin(distances[rows_idx, :][:, cols_idx])
                        i, j = np.unravel_index(min_idx, (len(rows_idx), len(cols_idx)))
                        row, col = rows_idx[i], cols_idx[j]
                        
                        # Si la distancia es razonable, asignar
                        if distances[row, col] < 100:  # umbral de distancia máxima
                            self.paths[object_ids[row]].append(detections[col])
                            self.disappeared[object_ids[row]] = 0
                            rows_idx.pop(i)
                            cols_idx.pop(j)
                        else:
                            # No hay buenas asignaciones restantes
                            break
                    else:
                        break
            
            # Manejar detecciones no asignadas (nuevos objetos)
            for col in cols_idx:
                self.paths[self.next_id] = [detections[col]]
                self.disappeared[self.next_id] = 0
                self.next_id += 1
            
            # Manejar objetos sin detecciones (desaparecidos)
            for row in rows_idx:
                obj_id = object_ids[row]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    del self.paths[obj_id]
                    del self.disappeared[obj_id]
        
        return self.paths

# --- Funciones auxiliares ---
def calculate_speed(points: List[Tuple[float, float]], time_diff: float) -> float:
    """Calcula la velocidad en cm/s entre dos puntos"""
    if len(points) < 2:
        return 0
    p1, p2 = points[-2], points[-1]
    distance = np.hypot(p2[0] - p1[0], p2[1] - p1[1])  # en cm
    return distance / time_diff  # cm/s

def draw_grid(frame, lines, color=(0, 255, 255), thickness=1):
    """Dibuja la cuadrícula detectada por HoughLines"""
    if lines is None:
        return
    
    h, w = frame.shape[:2]  
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(frame, (x1, y1), (x2, y2), color, thickness)

# --- Inicializaciones ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error al abrir el video.")
    exit()

# Inicializar modelo YOLO
print("Cargando modelo YOLOv8...")
model = YOLO("yolov8n.pt")  # Modelo base, ajusta según necesidades

# Configurar seguimiento de trayectorias
tracker = PathTracker()

# Variables para el sistema de coordenadas
pixel_per_cm = None
fps = cap.get(cv2.CAP_PROP_FPS)
dt = 1.0 / fps  # tiempo entre frames

# Variables para almacenar trayectorias
trajectories_cm = {}  # {id: [(x_cm, y_cm, frame_num), ...]}

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    # 1) Calcular escala (pix/cm) usando la cuadrícula
    lines = None
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
    
    # Dibujar cuadrícula si está disponible
    if lines is not None:
        draw_grid(frame, lines)
    
    # 2) Detectar patitos con YOLOv8
    results = model(frame, conf=CONFIDENCE_THRESH, iou=IOU_THRESH, classes=[0])  # clase 0 = persona, ajustar para patitos
    
    # Procesar las detecciones
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            
            # Solo procesar detecciones relevantes (ajustar según las clases que detecte tu modelo)
            if cls in [0, 16, 25]:  # persona, pájaro, o lo que corresponda a patitos
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                w, h = x2 - x1, y2 - y1
                area = w * h
                
                # Filtrar por área si es necesario
                if area > AREA_THRESHOLD:
                    detections.append((cx, cy, conf, cls))
                    
                    # Dibujar la detección en el frame
                    label = f"Patito {conf:.2f}"
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, label, (int(x1), int(y1 - 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # 3) Actualizar el seguimiento de trayectorias
    paths = tracker.update(detections)
    
    # 4) Calcular y mostrar trayectorias y velocidades
    if pixel_per_cm:
        for obj_id, points in paths.items():
            # Convertir últimos puntos a cm
            if len(points) >= 2:
                # Extraer centroide
                cx1, cy1 = points[-2][:2]
                cx2, cy2 = points[-1][:2]
                
                # Convertir a cm
                x1_cm, y1_cm = cx1 / pixel_per_cm, cy1 / pixel_per_cm
                x2_cm, y2_cm = cx2 / pixel_per_cm, cy2 / pixel_per_cm
                
                # Almacenar en trayectorias (en cm)
                if obj_id not in trajectories_cm:
                    trajectories_cm[obj_id] = []
                trajectories_cm[obj_id].append((x2_cm, y2_cm, frame_count))
                
                # Calcular velocidad
                speed = calculate_speed([(x1_cm, y1_cm), (x2_cm, y2_cm)], dt)
                
                # Dibujar línea de trayectoria
                cv2.line(frame, (int(cx1), int(cy1)), (int(cx2), int(cy2)), (0, 0, 255), 2)
                
                # Mostrar ID y velocidad
                cv2.putText(frame, f"ID: {obj_id}, v={speed:.1f} cm/s", 
                            (int(cx2), int(cy2 - 20)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Mostrar frame
    cv2.imshow('Seguimiento de Patitos con YOLOv8', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# --- Gráfica final de trayectorias (en cm) ---
if pixel_per_cm and trajectories_cm:
    plt.figure(figsize=(12, 8))
    
    # Graficar todas las trayectorias
    for obj_id, traj in trajectories_cm.items():
        if len(traj) < 2:
            continue
            
        xs, ys, _ = zip(*traj)
        plt.plot(xs, ys, '-o', label=f'Patito {obj_id}')
        
        # Dibujar flechas de dirección
        for i in range(len(xs)-1):
            dx, dy = xs[i+1]-xs[i], ys[i+1]-ys[i]
            plt.arrow(xs[i], ys[i], dx, dy,
                      head_width=0.5, length_includes_head=True, alpha=0.6)
    
    plt.gca().invert_yaxis()  # Invertir eje Y para coincidir con coordenadas de imagen
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')
    plt.title('Trayectorias y Direcciones de Patitos (YOLOv8)')
    plt.grid(True)
    plt.legend()
    
    # Calcular velocidades medias
    plt.figure(figsize=(12, 6))
    for obj_id, traj in trajectories_cm.items():
        if len(traj) < 3:
            continue
            
        speeds = []
        times = []
        xs, ys, frames = zip(*traj)
        
        for i in range(1, len(traj)):
            dx = xs[i] - xs[i-1]
            dy = ys[i] - ys[i-1]
            df = frames[i] - frames[i-1]
            
            if df > 0:  # evitar división por cero
                dist = np.hypot(dx, dy)
                time = df / fps
                speed = dist / time
                speeds.append(speed)
                times.append(frames[i] / fps)  # tiempo en segundos
        
        if speeds:
            plt.plot(times, speeds, '-o', label=f'Patito {obj_id}')
    
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Velocidad (cm/s)')
    plt.title('Velocidad vs Tiempo')
    plt.grid(True)
    plt.legend()
    
    plt.show()
else:
    print("No se pudieron graficar trayectorias.")