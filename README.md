<<<<<<< HEAD
# 🐘 TuskAlert: Real-Time Elephant Detection with YOLOv8
A YOLO-based detection model to identify elephants near electric or natural fences. This model can be integrated with automated deterrents or local alert systems, helping to mitigate the human-elephant conflict in rural Sri Lanka **without harming elephants or damaging property**.

## 🚀 Features
- **Real-time object detection** using **YOLOv8**
- Supports **both CPU & GPU inference**
- **Auto-detects NVIDIA GPU** and runs on CUDA if available
- **Logs detections & triggers alerts when elephants are detected**
- **Works with videos, images, or live webcam feeds**

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

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt

6. **Download and prepare the dataset**
   * Download from the links in Dataset section (below) and place images in data/   
   * Ensure labelled annotations (.xml or .txt) are in data/yolo_labels/ 

   NOTE: Only 329 images from the given datasets were used to train the model.

7. **Run the model locally on a test video**
   ```bash
   python local_inference.py


## 📖 Model Training

This model was trained using YOLOv8 from Ultralytics.

1. **Training the Model**
   Run the following command to start training:
   ```bash
   python train.py --data data.yaml --epochs 50 --imgsz 640 --device auto

2. **Training Results**
   After training, the best model weights will be saved under:
   ```bash
   runs/detect/train/weights/best.pt

## 🏃 Running Inference

To detect elephants in images, run:
```bash
python local_inference.py --source path/to/image.jpg
```

To run a video, modify local_inference.py:
```bash
video_stream = cv2.VideoCapture("path/to/video.mp4")
```

The output will be saved in the same directory as:
* elephant_detected.jpg → Image with bounding boxes.
* detection_log.txt → Log of detected timestamps.

## 📊 Model Performance

| Metric  | Value |
| ------------- | ------------- |
| mAP@50  | 93.4%  |
| mAP@50-95  | 73.8%  |
| Precision | 95.0% |
| Recall | 88.7% |
| Inference Speed | ~145ms per frame (CPU) |

🚀 **With an NVIDIA GPU, inference is much faster (~30ms per frame)!**

## 📚 Dataset
This project uses the following Kaggle datasets:
* Wild Elephant Dataset -- Download it [here](https://www.kaggle.com/datasets/gunarakulangr/sri-lankan-wild-elephant-dataset)
* Asian vs African Elephants -- Download it [here](https://www.kaggle.com/datasets/vivmankar/asian-vs-african-elephant-image-classification)

## 📜 Scripts

The `scripts/` folder contains essential scripts for data preparation, training, and evaluation:

| **Script** | **Description** | **Usage** |
|------------|---------------|-----------|
| **`labelled_image_filter.py`** | Moves labeled images into a separate folder for easier dataset handling | `python scripts/labelled_image_filter.py` |
| **`xml_to_yolo.py`** | Converts `.xml` annotation labels (Pascal VOC) to YOLO format | `python scripts/xml_to_yolo.py` |
| **`train_validation_test_split.py`** | Splits dataset into **training**, **validation**, and **test** sets | `python scripts/train_validation_test_split.py` |
| **`model_perf_metrics.py`** | Evaluates model performance and prints **mAP, Precision, Recall** | `python scripts/model_perf_metrics.py` |

---

### **How to Use These Scripts**

1. **Filter Labelled Images**
   Move only the labelled images into a dedicated folder:
   ```bash
   python scripts/labelled_image_filter.py
   ```
   This ensures only annotated images are used for training.

2. **Convert .xml labels to YOLO format**
   Convert Pascal VOC XML annotations into the required YOLO format:
   ```bash
   python scripts/xml_to_yolo.py
   ```
   This makes YOLO training possible since it requires .txt labels.

3. **Split Dataset into Train/Validation/Test**
   Organise filtered images into training, validation, and test folders:
   ```bash
   python scripts/train_validation_test_split.py
   ```
   This ensures a proper dataset split for training.

4. **Evaluate Model Performance**
   After training, calculate model performance metrics:
   ```bash
   python scripts/model_perf_metrics.py
   ```
   This script outputs mAP (Mean Average Precision), Precision, Recall, and Inference Speed.

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
>>>>>>> 6e60f07 (Removed large files and fixed history)
