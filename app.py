import gradio as gr
import cv2
from ultralytics import YOLO
import torch

# Auto-detect GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Safe Model Loading
try:
    model = YOLO("runs/detect/train6/weights/best.pt")  # Load YOLO model
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    model = None  # Prevents app from crashing

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

# Fix the Gradio `Video` input issue
iface = gr.Interface(
    fn=detect_objects,
    inputs=gr.Video(format="mp4"),  # ✅ Corrected parameter
    outputs=gr.JSON(),
    title="🐘 TuskAlert: Real-Time Elephant Detection",
    description="Upload a video to detect elephants in real-time."
)

iface.launch()
