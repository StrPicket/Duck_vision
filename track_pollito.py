import cv2
<<<<<<< HEAD
import math

VIDEO       = "video_original.mp4"
CM_PER_PX   = 0.2353      # escala: cm/píxel
EXPAND_FACTOR = 1.5       # amplía ROI inicial
MIN_MOVE_CM = 0.5         # cm mínimo para contar movimiento

def create_csrt():
    return cv2.legacy.TrackerCSRT_create()

roi_pts = []
def on_click(e,x,y,f,p):
    global roi_pts
    if e==cv2.EVENT_LBUTTONDOWN and len(roi_pts)<2:
        roi_pts.append((x,y))
        cv2.circle(p,(x,y),5,(0,0,255),-1)
        cv2.imshow("Define ROI", p)

def main():
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print("No puedo abrir video"); return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = int(1000/fps)

    ret, frame = cap.read()
    if not ret:
        print("Error leyendo video"); return
    H, W = frame.shape[:2]

    # 1) Selección ROI con 2 clicks
    disp = frame.copy()
    cv2.imshow("Define ROI", disp)
    cv2.setMouseCallback("Define ROI", on_click, disp)
    print("→ Haz 2 clicks: sup‑izq e inf‑der de tu pollito")
    while len(roi_pts) < 2:
        if cv2.waitKey(1)&0xFF == ord('q'):
            print("Cancelado"); return
    cv2.destroyAllWindows()

    (x1,y1),(x2,y2) = roi_pts
    x, y = min(x1,x2), min(y1,y2)
    w, h = abs(x2-x1), abs(y2-y1)
    # 2) Expande ROI
    cx, cy = x+w/2, y+h/2
    w = int(w*EXPAND_FACTOR); h = int(h*EXPAND_FACTOR)
    x = max(0, int(cx-w/2)); y = max(0, int(cy-h/2))
    w = min(w, W-x); h = min(h, H-y)
    print(f"ROI expandida: {(x,y,w,h)}")

    # 3) Init tracker
    tracker = create_csrt()
    tracker.init(frame, (x,y,w,h))

    # centro anterior
    prev_cx, prev_cy = x+w/2, y+h/2
    total_cm = 0.0

    while True:
        ret, frame = cap.read()
        if not ret: break

        ok, box = tracker.update(frame)
        if ok:
            x,y,w,h = box
            cx = x + w/2
            cy = y + h/2

            # distancia en cm
            dpx = math.hypot(cx - prev_cx, cy - prev_cy)
            dcm = dpx * CM_PER_PX

            # acumula solo si se movió
            if dcm >= MIN_MOVE_CM:
                total_cm += dcm
                prev_cx, prev_cy = cx, cy

            # dibujar ONLY when ok
            cv2.rectangle(frame, (int(x),int(y)),
                                (int(x+w),int(y+h)), (0,255,0),2)
            cv2.circle(frame,(int(cx),int(cy)),4,(0,0,255),-1)
            estado = "OK"
        else:
            # NO re-detectar: mantenemos prev_cx, prev_cy
            estado = "Lost"

        # feedback
        cv2.putText(frame, f"Estado: {estado}", (10,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0,255,0) if ok else (0,0,255), 2)
        cv2.putText(frame, f"Dist: {total_cm:.1f} cm", (10,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("CSRT Solo Mi Pollito", frame)
        if cv2.waitKey(delay)&0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n=== Distancia TOTAL: {total_cm:.2f} cm ===")
=======
import numpy as np
import math

VIDEO_PATH  = "video_original.mp4"
CM_PER_PX   = 0.2353   # escala: cm por píxel
MAX_CORNERS = 100      # max de puntos en la ROI
QUALITY     = 0.3      # calidad mínima de esquinas
MIN_DIST    = 7        # distancia mínima entre esquinas
ROI_MARGIN  = 10       # px de margen al actualizar la caja
MIN_POINTS  = 5        # puntos mínimos para considerar tracking válido
LK_PARAMS   = dict(winSize=(15,15),
                   maxLevel=3,
                   criteria=(cv2.TERM_CRITERIA_EPS |
                             cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("No se puede abrir el video"); return

    # FPS / delay
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = int(1000/fps)

    # Primer frame
    ret, first = cap.read()
    if not ret:
        print("Error leyendo primer frame"); return
    first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)

    # Selección manual de ROI
    x,y,w,h = cv2.selectROI("Marca ROI (pollito)", first, showCrosshair=True)
    cv2.destroyAllWindows()
    if w==0 or h==0:
        print("ROI inválida"); return

    # Detectar esquinas dentro de la ROI
    mask0 = np.zeros_like(first_gray)
    mask0[y:y+h, x:x+w] = 255
    p0 = cv2.goodFeaturesToTrack(first_gray, mask=mask0,
                                 maxCorners=MAX_CORNERS,
                                 qualityLevel=QUALITY,
                                 minDistance=MIN_DIST)
    if p0 is None or len(p0) < MIN_POINTS:
        print("Pocos puntos de interés en ROI"); return

    # Centro inicial
    pts = p0.reshape(-1,2)
    prev_cx, prev_cy = np.mean(pts, axis=0)

    total_px = 0.0
    total_cm = 0.0

    # Bucle
    old_gray = first_gray
    bbox = (x,y,w,h)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Optical flow
        p1, st, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **LK_PARAMS)
        if p1 is None or st is None:
            cv2.putText(frame, "¡Perdido!", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("OptFlow BBox", frame)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
            continue

        # Filtrar válidos y dentro de la caja ampliada
        good = p1[st.flatten()==1].reshape(-1,2)
        if len(good) < MIN_POINTS:
            cv2.putText(frame, "Pocos puntos", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("OptFlow BBox", frame)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
            continue

        # Calcular nuevo bounding box alrededor de good
        xs, ys = good[:,0], good[:,1]
        min_x, max_x = int(xs.min())-ROI_MARGIN, int(xs.max())+ROI_MARGIN
        min_y, max_y = int(ys.min())-ROI_MARGIN, int(ys.max())+ROI_MARGIN
        # Clamp a bordes
        min_x, min_y = max(0,min_x), max(0,min_y)
        max_x, max_y = min(frame.shape[1],max_x), min(frame.shape[0],max_y)
        bbox = (min_x, min_y, max_x-min_x, max_y-min_y)

        # Centro actual
        cx, cy = xs.mean(), ys.mean()
        # Distancia de movimiento
        dx = cx - prev_cx
        dy = cy - prev_cy
        step = math.hypot(dx, dy)
        total_px += step
        total_cm += step * CM_PER_PX
        prev_cx, prev_cy = cx, cy

        # Dibujar resultados
        cv2.rectangle(frame, (bbox[0],bbox[1]),
                           (bbox[0]+bbox[2],bbox[1]+bbox[3]), (0,255,0), 2)
        for p in good:
            cv2.circle(frame, (int(p[0]),int(p[1])), 3, (0,255,0), -1)
        cv2.circle(frame, (int(cx),int(cy)), 5, (0,0,255), -1)
        cv2.putText(frame, f"Dist: {total_cm:.1f} cm", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("OptFlow BBox", frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

        # Preparar siguiente iteración
        old_gray = frame_gray
        # Re-definir p0 como los buenos dentro de la nueva caja
        p0 = good.reshape(-1,1,2)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n=== Total recorrido: {total_cm:.2f} cm ({total_px:.1f} px) ===")
>>>>>>> 1d27a06e777a873e018a89b492d17afe07d4cac0

if __name__ == "__main__":
    main()
