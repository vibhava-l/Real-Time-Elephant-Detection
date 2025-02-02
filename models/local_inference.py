import cv2
from ultralytics import YOLO
import winsound
import time
import torch

# Auto-detect GPU (if available), otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")

# Load the trained YOLOv8 model
model = YOLO('F:/Real-Time-Elephant-Detection/runs/detect/train6/weights/best.pt')  # Path to the trained weights

# Initialise the video stream (0 for webcam or provide the path to the video file)
video_stream = cv2.VideoCapture('F:\Real-Time-Elephant-Detection\Clever elephants climb over electric fence in Sri Lanka.mp4')  # Replace 0 with 'path_to_video.mp4' for a video file (or vice-versa)

# Function to trigger an alert
def trigger_alert():
    print('ALERT: Elephant detected!')
    winsound.Beep(1000, 2000)  # Beep for 2 seconds
    # Append detection events to a log file
    with open('detection_log.txt', 'a') as log:
        log.write('Elephant detected at time {}\n'.format(time.ctime()))

# Check if the video stream is opened successfully
if not video_stream.isOpened():
    print('Error occurred while opening the video stream. Exiting...')
    exit()

elephant_detected = False  # Tracks if an elephant is currently detected
last_alert_time = 0  # Timestamp of the last alert
alert_cooldown = 10  # Cooldown period in seconds

while True:
    # Capture the video frame-by-frame
    ret, frame = video_stream.read()
    if not ret:
        print('Video stream ended or error occurred.')
        break

    # Resize frame for faster processing
    frame = cv2.resize(frame, (640, 480))

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Check if an elephant is detected
    if results[0].boxes:  # Checks if there are any detections in the frame
        current_time = time.time()
        if not elephant_detected or (current_time - last_alert_time > alert_cooldown):  # Trigger alert only if detection is new
            trigger_alert()  # Call the alert function
            last_alert_time = current_time  # Update timestamp of the last alert
            elephant_detected = True  # Update state
        cv2.imwrite('elephant_detected.jpg', frame)
    else:
        elephant_detected = False  # Reset state when no elephants are detected

    # Visualise detections on the frame
    annotated_frame = results[0].plot()

    # Display the resulting frame
    cv2.imshow('YOLOv8 Real-time Inference', annotated_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close the OpenCV windows
video_stream.release()
cv2.destroyAllWindows()