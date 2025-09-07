from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")

# Force GPU
model.train(
    data="/content/drive/MyDrive/fruit_dataset/data.yaml", # 👈 change to your own file path
    epochs=50,
    imgsz=640,
    batch=16,
    workers=2,
    device=0,   # 👈 force GPU (0 = first GPU, 'cpu' = CPU)
    project="/content/drive/MyDrive/runs", # 👈 change to your own file path
    name="fruit_detection"
)