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

# Function to process video and detect elephants
def detect_objects(video_path):
    if model is None:
        return "❌ Model failed to load. Check logs for details.", "", ""

    cap = cv2.VideoCapture(video_path)  # Open video file
    detections = []  # Store detected objects
    frame_list = []  # Store processed frames

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Define codec for saving video
    output_path = "output.mp4"
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get FPS of input video
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get frame dimensions
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))  # Create output video file

    alert_message = "No elephants detected."  # Default alert message

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, device=device)  # Run YOLO model on each frame
        elephant_detected = False  # Track if an elephant was found in this frame

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                conf = box.conf[0].item()  # Confidence score

                # Draw the bounding box and confidence score
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Store detection details
                detections.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": conf})

                # Update alert message if an elephant is detected
                if conf > 0.5:  # Adjust confidence threshold as needed
                    elephant_detected = True

        if elephant_detected:
            alert_message = "🚨 Elephant detected! Stay alert!"

    cap.release()
    out.release()

    return output_path, detections, alert_message  # Return updated alert message

# Define the Gradio interface
iface = gr.Interface(
    fn=detect_objects,  # Function to be called when a user uploads a video
    inputs=gr.Video(label="Upload a video"),
    outputs=[
        gr.Video(label="Processed Video"),
        gr.JSON(label="Detections Log"),
        gr.Textbox(label="Elephant Alert", interactive=False),  # Display alert messages
    ],
    title="🐘 TuskAlert: Real-Time Elephant Detection",
    description="Upload a video to detect elephants and visualize detections in real-time. Alerts will display when elephants are detected.",
)

iface.launch()
