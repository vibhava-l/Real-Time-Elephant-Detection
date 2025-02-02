import gradio as gr
import cv2
from ultralytics import YOLO
import torch
import torch.serialization

# Auto-detect GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Allow loading of custom YOLO model safely
torch.serialization.add_safe_globals(["ultralytics.nn.tasks.DetectionModel"])

# Modify this line to explicitly set `weights_only=False`
model = YOLO("runs/detect/train6/weights/best.pt")

def detect_objects(video_path):
    cap = cv2.VideoCapture(video_path)  # Open video file
    detections = []  # Store detected objects

    while cap.isOpened():  # Loop through video frames
        ret, frame = cap.read()
        if not ret:  # If no frame is read, break the loop
            break

        results = model(frame, device=device)  # Run YOLO on the frame
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                conf = box.conf[0].item()  # Confidence score
                detections.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": conf})

    cap.release()  # Close the video file
    return detections

# Define the Gradio interface
iface = gr.Interface(
    fn=detect_objects,  # Function to be called when a user uploads a video
    inputs=gr.Video(type="filepath"),  # Input: User uploads a video file from their system
    outputs=gr.JSON(),  # Output: JSON containing detected elephant bounding boxes & confidence scores
    title="🐘 TuskAlert: Real-Time Elephant Detection",
    description="Upload a video to detect elephants and prevent conflicts in real-time."
)

# Launch the Gradio interface (starts a local server to serve the app, or deploys it on Hugging Face if pushed)
iface.launch()
