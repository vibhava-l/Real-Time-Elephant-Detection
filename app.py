import gradio as gr
import cv2
from ultralytics import YOLO
import torch
import os
import platform
import subprocess
import tempfile
import time
from pydub import AudioSegment
from pydub.playback import play
from gtts import gTTS

# Auto-detect GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Safe Model Loading
try:
    model = YOLO("runs/detect/train6/weights/best.pt")  # Load YOLO model
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    model = None  # Prevents app from crashing

# Function to play sound alert when elephant is detected (with cooldown)
last_alert_time = 0  # Global variable for cooldown

def play_alert():
    global last_alert_time
    if time.time() - last_alert_time < 5:  # 5-second cooldown
        return

    last_alert_time = time.time()  # Update last alert time

    try:
        tts = gTTS("Elephant detected! Stay alert!", lang="en")
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_audio.name)

        sound = AudioSegment.from_file(temp_audio.name, format="mp3")
        play(sound)
        print("🔊 Elephant alert sounded!")

    except Exception as e:
        print(f"❌ Error generating alert: {e}")

# Function to process video and detect elephants
def detect_objects(video_path):
    if model is None:
        yield "❌ Model failed to load. Check logs for details.", None

    cap = cv2.VideoCapture(video_path)  # Open video file
    if not cap.isOpened():
        yield "❌ Could not open video file.", None

    detections = []  # Store detected objects
    frame_count = 0
    frame_skip = 5  # Process every 5th frame (5x speed-up)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

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

                    # Play beep sound when elephant is detected
                    if conf > 0.5:  # Adjust confidence threshold as needed
                        play_alert()

            # ✅ Convert frame to a file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            cv2.imwrite(temp_file.name, frame)  # Save frame as a temporary image
            yield temp_file.name, detections  # **Return file path, NOT bytes**

        frame_count += 1

    cap.release()

# Define the Gradio interface
iface = gr.Interface(
    fn=detect_objects,
    inputs=gr.Video(label="Upload a video"),
    outputs=[gr.Image(label="Live Processed Video"), gr.JSON(label="Detections Log")],
    title="🐘 TuskAlert: Real-Time Elephant Detection",
    description="Upload a video to detect elephants and visualize detections in real-time. A voice alert will play when an elephant is detected."
)

iface.launch()
