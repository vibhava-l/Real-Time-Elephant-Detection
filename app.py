import gradio as gr
import cv2
from ultralytics import YOLO
import torch
import os
import platform
import subprocess
import tempfile

# Try importing playsound for audio alerts
try:
    from playsound import playsound
    PLAYSOUND_AVAILABLE = True
except ImportError:
    PLAYSOUND_AVAILABLE = False

# Auto-detect GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Safe Model Loading
try:
    model = YOLO("runs/detect/train6/weights/best.pt")  # Load YOLO model
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    model = None  # Prevents app from crashing

# Function to play sound alert when elephant is detected
def play_alert():
    alert_file = "alert_sound.mp3"  # Ensure this file exists in repo
    if platform.system() == "Windows":
        import winsound
        winsound.PlaySound(alert_file, winsound.SND_FILENAME)
    else:
        # Use ffplay to play the sound on Linux (Hugging Face Spaces)
        subprocess.Popen(["ffplay", "-nodisp", "-autoexit", alert_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Function to process video and detect elephants
def detect_objects(video_path):
    if model is None:
        return "❌ Model failed to load. Check logs for details."

    cap = cv2.VideoCapture(video_path)  # Open video file
    if not cap.isOpened():
        return "❌ Could not open video file."

    detections = []  # Store detected objects

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))  
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  

    # Create temporary output video path
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, "output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  
    out = cv2.VideoWriter(output_path, fourcc, fps, (width // 2, height // 2))  # Reduce resolution for faster output

    frame_count = 0
    frame_skip = 5  # Process every 5th frame (5x speed-up)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:  # Skip frames for faster processing
            resized_frame = cv2.resize(frame, (320, 320))  # Resize for faster inference
            results = model(resized_frame, device=device)

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                    conf = box.conf[0].item()  # Confidence score

                    # Scale coordinates back to original resolution
                    x1, x2 = int(x1 * width / 320), int(x2 * width / 320)
                    y1, y2 = int(y1 * height / 320), int(y2 * height / 320)

                    # Draw the bounding box and confidence score
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Store detection details
                    detections.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": conf})

                    # Play beep sound when elephant is detected
                    if conf > 0.5:  # Adjust confidence threshold as needed
                        play_alert()

        out.write(cv2.resize(frame, (width // 2, height // 2)))  # Save processed frame

        frame_count += 1

    cap.release()
    out.release()

    return output_path, detections

# Define the Gradio interface
iface = gr.Interface(
    fn=detect_objects,  # Function to be called when a user uploads a video
    inputs=gr.Video(label="Upload a video"),
    outputs=[gr.Video(label="Processed Video"), gr.JSON(label="Detections Log")],
    title="🐘 TuskAlert: Real-Time Elephant Detection",
    description="Upload a video to detect elephants and visualize detections in real-time. A beep sound will play when an elephant is detected."
)

iface.launch()
