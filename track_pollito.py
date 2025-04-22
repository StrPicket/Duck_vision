import cv2
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

if __name__ == "__main__":
    main()
