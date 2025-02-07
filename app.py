import gradio as gr
import cv2
from ultralytics import YOLO
import torch
import os
import platform
import subprocess
from gtts import gTTS
from pydub import AudioSegment

# Auto-detect GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Safe Model Loading
try:
    model = YOLO("runs/detect/train6/weights/best.pt")  # Load YOLO model
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    model = None  # Prevents app from crashing

# Function to generate an alert sound file
def generate_alert():
    alert_text = "Warning! Elephant detected!"
    tts = gTTS(alert_text)
    alert_file = "/tmp/elephant_alert.wav"
    tts.save(alert_file)

    # Convert to MP3
    sound = AudioSegment.from_file(alert_file, format="wav")
    sound.export("/tmp/elephant_alert.mp3", format="mp3")

    print("🔊 Elephant alert generated!")
    return "/tmp/elephant_alert.mp3"

# Function to process video and detect elephants
def detect_objects(video_path):
    if model is None:
        return "❌ Model failed to load. Check logs for details."

    cap = cv2.VideoCapture(video_path)  # Open video file
    detections = []  # Store detected objects
    frame_list = []  # Store processed frames

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Define codec for saving video
    output_path = "output.mp4"
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get FPS of input video

    # Reduce output video resolution
    width, height = 640, 480  # Resized resolution
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))  # Create output video file

    elephant_detected = False  # Track if any elephant was found

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (width, height))  # Resize frame for faster processing
        results = model(frame, device=device)  # Run YOLO model on each frame

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                conf = box.conf[0].item()  # Confidence score

                # Draw bounding box and confidence score
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Store detection details
                detections.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": conf})

                # Trigger alert sound only once
                if conf > 0.5 and not elephant_detected:
                    elephant_detected = True

        out.write(frame)  # Save frame with detections

    cap.release()
    out.release()

    # Generate alert sound file
    alert_audio_path = None
    if elephant_detected:
        alert_audio_path = generate_alert()

    return output_path, alert_audio_path, detections

# Define the Gradio interface
iface = gr.Interface(
    fn=detect_objects,  # Function to be called when a user uploads a video
    inputs=gr.Video(label="Upload a video"),
    outputs=[
        gr.Video(label="Processed Video"),
        gr.Audio(label="Alert Sound (Downloadable)"),
        gr.JSON(label="Detections Log")
    ],
    title="🐘 TuskAlert: Real-Time Elephant Detection",
    description="Upload a video to detect elephants and visualize detections in real-time. An alert sound will be generated when an elephant is detected."
)

iface.launch()
