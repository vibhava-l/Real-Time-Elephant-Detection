---
title: TuskAlert Elephant Detection
emoji: 🐘
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 5.14.0
app_file: app.py
pinned: false
---
# 🐘 TuskAlert: Real-Time Elephant Detection with YOLOv8
A YOLOv8-based detection model for identifying elephants near electric or natural fences. This project aims to mitigate human-elephant conflicts in rural areas like Sri Lanka by providing a **non-intrusive, real-time alerting system**. The system leverages **Ultralytics YOLOv8** for object detection and integrates with Gradio to offer a user-friendly web interface.

## 🚀 Features
- **Real-time Detection**: Detect elephants in uploaded video files or live camera feeds.
- **Web Interface**: A clean and simple Gradio UI for video uploads and JSON-based output.
- **Dynamic Inference Handling**: Supports both CPU & GPU execution for efficient inference.
- **Extensive logging**: Errors and detections are logged for debugging and traceability.

---

## 🔧 Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/vibhava-l/Real-Time-Elephant-Detection.git
   cd Real-Time-Elephant-Detection

2. **Create a virtual environment**
   ```bash
   python -m venv elephant_fence_env
   source elephant_fence_env/bin/activate  # On macOS/Linux
   elephant_fence_env\Scripts\activate  # On Windows

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt

---

## 🖥 Run the application locally**
Launch the Gradio interface locally:
```bash
python app.py
```
By default, the app runs on http://locahost:7860. Open this URL in your browser to access the application.

## 🐘 Using the Gradio Interface

1. Upload a video of the target area.
2. The system will process the video to detect elephants.
3. Results will be displayed in JSON format, listing bounding box coordinates and confidence scores for detected objects.

Example Output:

```
[
    {"x1": 34, "y1": 56, "x2": 123, "y2": 178, "confidence": 0.95},
    {"x1": 223, "y1": 89, "x2": 289, "y2": 200, "confidence": 0.87}
]
```

---

## 🧪 Testing

To test locally:

* Ensure a valid best.pt model file is available in the runs/detect/train6/weights/ directory.
* Upload a sample video via the Gradio interface and verify the detections.

## 🤝 Contributing

Want to improve this project? Follow these steps:
 1. Fork the repository
 2. Create a feature branch (```git checkout -b feature-name ```)
 3. Commit changes (```git commit -m "Add new feature" ```)
 4. Push the branch (```git push origin feature-name ```)
 5. Open a Pull Request

## 📩 Contact

For questions or collaboration, contact me at:

📧 Email: [vibhava.leelaratna@gmail.com]

🌍 GitHub: github.com/vibhava-l