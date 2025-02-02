from ultralytics import YOLO

# Load pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # Lightweight model

# Train the model
model.train(data='models/data.yaml', epochs=50, imgsz=640)

# Save the best weights after training
best_weights = 'runs/train/exp/weights/best.pt'
print(f'Training complete. Best model saved as {best_weights}')