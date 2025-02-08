import gradio as gr
import cv2
from ultralytics import YOLO
import torch
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

# Function to process video and detect elephants (Real-Time Streaming with Alerts)
def detect_objects(video_path):
    if model is None:
        yield "❌ Model failed to load. Check logs for details.", None, "❌ Model not loaded"

    cap = cv2.VideoCapture(video_path)  # Open video file
    if not cap.isOpened():
        yield "❌ Could not open video file.", None, "❌ Video file error"

    detections = []  # Store detected objects
    frame_count = 0
    frame_skip = 5  # Process every 5th frame for faster inference
    alert_message = "No elephants detected."  # Default alert message

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:  # Skip frames for faster processing
            resized_frame = cv2.resize(frame, (320, 320))  # Resize for speed
            results = model(resized_frame, device=device)

            elephant_detected = False  # Track if an elephant was found in this frame

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

                    # Update alert message when an elephant is detected
                    if conf > 0.5:  # Adjust confidence threshold as needed
                        elephant_detected = True

            if elephant_detected:
                alert_message = "🚨 Elephant detected! Stay alert!"
            else:
                alert_message = "No elephants detected."

            # **✅ Convert frame to a file**
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            cv2.imwrite(temp_file.name, frame)  # Save frame as a temporary image
            yield temp_file.name, alert_message, detections  # **Return file path, NOT bytes**

        frame_count += 1

    cap.release()

# Define the Gradio interface
iface = gr.Interface(
    fn=detect_objects,  # Function to be called when a user uploads a video
    inputs=gr.Video(label="Upload a video"),
    outputs=[
        gr.Image(label="Live Processed Video"),
        gr.Textbox(label="Elephant Alert", interactive=False),  # Live alert message
        gr.JSON(label="Detections Log"),
    ],
    title="🐘 TuskAlert: Real-Time Elephant Detection",
    description="Upload a video to detect elephants and visualize detections in real-time. Alerts will display when elephants are detected.",
)

iface.queue()  # Enable queuing system
iface.launch()
