import gradio as gr
import cv2
import torch
import threading
from ultralytics import YOLO
import tempfile

# Auto-detect GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Safe Model Loading
try:
    model = YOLO("runs/detect/train6/weights/best.pt").to(device)  # Load YOLO model on GPU if available
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    model = None  # Prevents app from crashing

# Process Video with Optimizations
def detect_objects(video_path):
    if model is None:
        yield "❌ Model failed to load. Check logs for details.", None, "❌ Model not loaded"

    cap = cv2.VideoCapture(video_path)  # Open video file
    if not cap.isOpened():
        yield "❌ Could not open video file.", None, "❌ Video file error"

    frame_skip = 5  # Skip every 5 frames to improve speed
    frame_count = 0
    alert_message = "No elephants detected."

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            resized_frame = cv2.resize(frame, (320, 320))  # Resize for faster inference

            # Run inference in a separate thread to prevent blocking
            detection_thread = threading.Thread(target=process_frame, args=(resized_frame, frame))
            detection_thread.start()
            detection_thread.join()  # Wait for thread to finish

        frame_count += 1

    cap.release()

def process_frame(resized_frame, original_frame):
    results = model(resized_frame, device=device)

    elephant_detected = False

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0].item()

            # Scale coordinates back to original resolution
            x1, x2 = int(x1 * original_frame.shape[1] / 320), int(x2 * original_frame.shape[1] / 320)
            y1, y2 = int(y1 * original_frame.shape[0] / 320), int(y2 * original_frame.shape[0] / 320)

            # Draw bounding box
            cv2.rectangle(original_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(original_frame, f"{conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if conf > 0.5:
                elephant_detected = True

    alert_message = "🚨 Elephant detected! Stay Alert! 🚨" if elephant_detected else "No elephants detected."

    # Save frame as temporary image
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(temp_file.name, original_frame)

    yield temp_file.name, alert_message

# Live Webcam Support
def live_webcam():
    cap = cv2.VideoCapture(0)  # Open webcam (0 for default webcam)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, device=device)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        yield frame  # Stream back processed frames

# RTSP Stream Support (CCTV/IP Camera)
def live_rtsp(rtsp_url="rtsp://192.168.1.10:554/live"):
    cap = cv2.VideoCapture(rtsp_url)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, device=device)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        yield frame

# Gradio Interface
iface = gr.Interface(
    fn=detect_objects,
    inputs=gr.Video(label="Upload a video"),
    outputs=[
        gr.Image(label="Live Processed Video"),
        gr.Textbox(label="Elephant Alert", interactive=False),
    ],
    title="🐘 TuskAlert: Real-Time Elephant Detection",
    description="Upload a video to detect elephants and visualize detections in real-time."
)

# Add Live Video Support
webcam_iface = gr.Interface(fn=live_webcam, inputs=[], outputs=gr.Image(label="Live Webcam Feed"), live=True)
rtsp_iface = gr.Interface(fn=live_rtsp, inputs=[gr.Textbox(label="RTSP Stream URL")], outputs=gr.Image(label="Live RTSP Feed"), live=True)

# **Launch All Interfaces**
gr.TabbedInterface([iface, webcam_iface, rtsp_iface], ["Upload Video", "Live Webcam", "RTSP Camera"]).launch(share=True)
