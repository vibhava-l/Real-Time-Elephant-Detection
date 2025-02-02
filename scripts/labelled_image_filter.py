import os
import shutil

# Paths
image_folder = "F:/Real-Time-Elephant-Detection/data"
label_folder = "F:/Real-Time-Elephant-Detection/data/yolo_labels"
output_image_folder = "F:/Real-Time-Elephant-Detection/incomplete_data"
output_label_folder = "F:/Real-Time-Elephant-Detection/incomplete_data/yolo_labels"

# Create output directories if they don't exist
if not os.path.exists(output_image_folder):
    os.makedirs(output_image_folder, exist_ok=True)
if not os.path.exists(output_label_folder):
    os.makedirs(output_label_folder, exist_ok=True)

# Filter out images with labels
for label_file in os.listdir(label_folder):
    if label_file.endswith(".txt"):
        # Get corresponding image file
        image_file = label_file.replace(".txt", ".jpg")  # Adjust extension if necessary
        image_path = os.path.join(image_folder, image_file)
        label_path = os.path.join(label_folder, label_file)

        if os.path.exists(os.path.join(image_folder, image_file)):  # Ensure image exists
            # Copy image and label to output directory
            shutil.copy(image_path, output_image_folder)
            shutil.copy(label_path, output_label_folder)

print(f"Filtered images and labels copied to {output_image_folder} and {output_label_folder}")