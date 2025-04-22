import cv2

# ------------------------------------------------------------------
# Usamos el video ya descargado con yt-dlp (debe estar en la misma carpeta)
video_path = "video_original.mp4"

# ------------------------------------------------------------------
# 1. Abrir el video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: no se pudo abrir '{video_path}'")
    exit()

# ------------------------------------------------------------------
# 2. Preparar VideoWriter para la salida
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("video_canny.avi", fourcc, fps, (width, height))

# ------------------------------------------------------------------
# 3. Procesamiento frame a frame
while True:
    ret, frame = cap.read()
    if not ret:
        break  # fin del video

    # Escala de grises
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Desenfoque Gaussiano
    blur = cv2.GaussianBlur(gris, (5, 5), 0)
    # Detección de bordes Canny
    edges = cv2.Canny(blur, 75, 75)
    # Convertir a BGR para que VideoWriter acepte 3 canales
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Mostrar en pantalla
    cv2.imshow("Canny Video", edges)
    # Escribir el frame procesado
    out.write(edges_bgr)

    # Salir al presionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ------------------------------------------------------------------
# 4. Liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()

print("Procesamiento completo. Video guardado como 'video_canny.avi'.")
