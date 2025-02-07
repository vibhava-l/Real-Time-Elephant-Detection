import gradio as gr
import cv2
from ultralytics import YOLO
import torch
import os
import platform
import subprocess
import tempfile

# Auto-detect GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Safe Model Loading
try:
    model = YOLO("runs/detect/train6/weights/best.pt")  # Load YOLO model
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    model = None  # Prevents app from crashing

# Increase FFmpeg Read Attempts to Avoid Video Read Errors
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "10000"


# ✅ **1️⃣ Function to Preprocess Video (Fix Multi-Stream Issues)**
def preprocess_video(video_path):
    """ Convert video to a compatible format (H.264, single-stream, no audio) """
    converted_video_path = "processed_video.mp4"
    command = [
        "ffmpeg", "-i", video_path, "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-an", converted_video_path, "-y"
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return converted_video_path


# ✅ **2️⃣ Function to Process Video and Detect Elephants**
def detect_objects(video_path):
    if model is None:
        yield "❌ Model failed to load. Check logs for details.", None, "❌ Model not loaded"

    # **Preprocess Video Before Processing**
    video_path = preprocess_video(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        yield "❌ Could not open video file.", None, "❌ Video file error"

    detections = []  # Store detected objects
    frame_count = 0
    frame_skip = 5  # Process every 5th frame for speed optimization

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Skipping a corrupt frame.")
            continue  # Skip broken frame and move to the next one

        if frame_count % frame_skip == 0:
            resized_frame = cv2.resize(frame, (320, 320))  # Resize for faster inference
            results = model(resized_frame, device=device)

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                    conf = box.conf[0].item()  # Confidence score

                    # Scale coordinates back to original resolution
                    x1, x2 = int(x1 * frame.shape[1] / 320), int(x2 * frame.shape[1] / 320)
                    y1, y2 = int(y1 * frame.shape[0] / 320), int(y2 * frame.shape[0] / 320)

                    # Draw the bounding box and confidence score
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Store detection details
                    detections.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": conf})

            # **✅ Convert frame to a file for Gradio Display**
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            cv2.imwrite(temp_file.name, frame)  # Save frame as a temporary image
            yield temp_file.name, detections  # **Return file path, NOT bytes**

        frame_count += 1

    cap.release()


# ✅ **3️⃣ Define the Gradio Interface**
iface = gr.Interface(
    fn=detect_objects,  # Function to be called when a user uploads a video
    inputs=gr.Video(label="Upload a video"),
    outputs=[gr.Image(label="Live Processed Video"), gr.JSON(label="Detections Log")],
    title="🐘 TuskAlert: Real-Time Elephant Detection",
    description="Upload a video to detect elephants and visualize detections in real-time."
)

iface.launch()
