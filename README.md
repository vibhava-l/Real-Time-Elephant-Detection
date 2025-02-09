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
TuskAlert is a **YOLOv8-based detection system** designed to **monitor elephant movement** near fences, reducing **human-elephant conflicts**. This project leverages **Ultralytics YOLOv8** for **real-time object detection** and integrates with **Gradio** to provide an easy-to-use web interface.

## 🚀 Features
- **Real-Time Detection** – Detect elephants in **uploaded videos** or **live camera feeds**.
- **Web Interface** – A user-friendly **Gradio UI** for easy interaction.
- **GPU & CPU Execution** – Supports **CUDA acceleration** for improved inference speed.
- **Live Alerts** – Provides **real-time warnings** when elephants are detected.
- **Scalable Deployment** – Hosted on **Hugging Face Spaces** for quick testing and **AWS EC2** for cloud deployment.

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

## 🚀 Deployment Options

### 🤗 1. Hugging Face Spaces

TuskAlert is deployed on Hugging Face Spaces for easy access.

To deploy manually:

```bash
gradio deploy
```

✅ **Pros**:
- Free hosting
- No server setup required
- Ideal for quick demos

### ☁ 2. AWS EC2 Deployment

For real-time processing and full control, deploy on an AWS EC2 instance.

Steps:
   1. Launch an EC2 instance (Ubuntu 20.04, t2.medium or higher).
   2. SSH into the instance:
      ```bash
      ssh -i "your-key.pem" ubuntu@your-ec2-ip
      ```
   3. Clone the repository:
      ```bash
      git clone https://github.com/vibhava-l/TuskAlert-Elephant-Detection.git
      cd TuskAlert-Elephant-Detection
      ```
   4. Set up the environment & run:
      ```bash
      python3 -m venv venv
      source venv/bin/activate
      pip install -r requirements.txt
      python app.py
      ```
   5. Access the live app via Gradio's public link.

✅ **Pros**:
- Better performance for large-scale inference
- More control over dependencies & environment
- Scalable with AWS infrastructure

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