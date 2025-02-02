from ultralytics import YOLO

model = YOLO("runs/detect/train6/weights/best.pt")  # Load trained model
results = model.val(data="F:/Real-Time-Elephant-Detection/models/data.yaml")
print(results)  # Prints detailed validation metrics
