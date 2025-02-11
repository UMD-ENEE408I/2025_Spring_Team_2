from ultralytics import YOLO
import torch
print(torch.cuda.is_available())  # Should print True

# Load YOLOv8 model (choose a small version to start)
model = YOLO("yolov8n.pt")

# Train on the TurtleBot dataset
model.train(
    data="data.yaml",
    epochs=5,  # Adjust as needed
    imgsz=194,
    batch=8,
    device="cpu"  # Use "cpu" if no GPU available
)
