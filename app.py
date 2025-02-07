import gradio as gr
import cv2
from ultralytics import YOLO
import torch
import os
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

def preprocess_video(video_path):
    """ Convert video to a compatible format (H.264, single-stream, no audio) """
    converted_video_path = "processed_video.mp4"
    command = [
        "ffmpeg", "-i", video_path, "-c:v", "libx264", "-crf", "28", "-preset", "ultrafast",
        "-an", converted_video_path, "-y"
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return converted_video_path


def detect_objects(video_path):
    if model is None:
        yield "❌ Model failed to load. Check logs for details.", None, "❌ Model not loaded"

    # Preprocess Video for Fast Processing
    video_path = preprocess_video(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        yield "❌ Could not open video file.", None, "❌ Video file error"

    detections = []
    frame_count = 0
    frame_skip = 10
    resize_dim = (224, 224)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Skipping a corrupt frame.")
            continue  # Skip broken frame

        if frame_count % frame_skip == 0:
            resized_frame = cv2.resize(frame, resize_dim)
            results = model([resized_frame], device=device)  # Batch Processing for faster inference

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                    conf = box.conf[0].item()  # Confidence score

                    # Scale back to original resolution
                    x1, x2 = int(x1 * frame.shape[1] / resize_dim[0]), int(x2 * frame.shape[1] / resize_dim[0])
                    y1, y2 = int(y1 * frame.shape[0] / resize_dim[1]), int(y2 * frame.shape[0] / resize_dim[1])

                    # Draw bounding box & confidence
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    detections.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": conf})

            # Save frame to temp file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            cv2.imwrite(temp_file.name, frame)
            yield temp_file.name, detections  # Faster frame updates

        frame_count += 1

    cap.release()


# Define the Gradio Interface
iface = gr.Interface(
    fn=detect_objects,
    inputs=gr.Video(label="Upload a video"),
    outputs=[gr.Image(label="Live Processed Video"), gr.JSON(label="Detections Log")],
    title="🐘 TuskAlert: Ultra-Fast Elephant Detection",
    description="Upload a video to detect elephants in real-time (optimized for speed)."
)

iface.launch()
