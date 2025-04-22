from ultralytics import YOLO

# Carga modelo base YOLOv8 nano (puedes usar yolov8s.pt si quieres más precisión)
model = YOLO('yolov8n.pt')

# Entrena con tu dataset
model.train(
    data='MiPatito.v1i.yolov8/data.yaml',  # Ruta al archivo YAML ya corregido
    epochs=50,
    imgsz=640,
    batch=8,
    name='mi_patito_detector',
    project='entrenamiento_patito'
)
