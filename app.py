import gradio as gr
import cv2
from ultralytics import YOLO
import torch

# Auto-detect GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Safe model loading with PyTorch 2.6+
try:
    torch.serialization.add_safe_globals(["ultralytics.nn.tasks.DetectionModel"])
    model = torch.load("runs/detect/train6/weights/best.pt", map_location="cpu", weights_only=False)
    print("✅ Model loaded with PyTorch safe mode")
except Exception as e:
    print(f"⚠️ Error loading best.pt: {e}")

    try:
        print("🔄 Retrying with YOLO model loader...")
        model = YOLO("runs/detect/train6/weights/best.pt")  # Fallback YOLO loading
    except Exception as e2:
        print(f"❌ Fallback YOLO loading failed: {e2}")
        model = None  # Prevent crash

def detect_objects(video_path):
    if model is None:
        return "❌ Model failed to load. Check logs for details."

    cap = cv2.VideoCapture(video_path)
    detections = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, device=device)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                detections.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": conf})

    cap.release()
    return detections

# Define Gradio UI
iface = gr.Interface(
    fn=detect_objects,
    inputs=gr.Video(type="filepath"),
    outputs=gr.JSON(),
    title="🐘 TuskAlert: Real-Time Elephant Detection",
    description="Upload a video to detect elephants in real-time."
)

iface.launch()
