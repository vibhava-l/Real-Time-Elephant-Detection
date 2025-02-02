import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
filtered_image_folder = "F:/Real-Time-Elephant-Detection/incomplete_data"
filtered_label_folder = "F:/Real-Time-Elephant-Detection/incomplete_data/yolo_labels"
output_folder = "F:/Real-Time-Elephant-Detection/incomplete_data"

# Get all filtered images and labels
images = sorted([f for f in os.listdir(filtered_image_folder) if f.endswith('.jpg')])
labels = sorted([f for f in os.listdir(filtered_label_folder) if f.endswith('.txt')])

# Ensure images and labels match
assert len(images) == len(labels), "Number of images and labels do not match"

# Split into train, validation, and test sets
train_images, temp_images, train_labels, temp_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
val_images, test_images, val_labels, test_labels = train_test_split(temp_images, temp_labels, test_size=0.5, random_state=42)

# Helper function to move files
def move_files(file_list, source_folder, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    for f in file_list:
        shutil.copy(os.path.join(source_folder, f), os.path.join(dest_folder, f))

# Move files to train, validation, and test folders
move_files(train_images, filtered_image_folder, os.path.join(output_folder, 'images/train'))
move_files(train_labels, filtered_label_folder, os.path.join(output_folder, 'labels/train'))
move_files(val_images, filtered_image_folder, os.path.join(output_folder, 'images/val'))
move_files(val_labels, filtered_label_folder, os.path.join(output_folder, 'labels/val'))
move_files(test_images, filtered_image_folder, os.path.join(output_folder, 'images/test'))
move_files(test_labels, filtered_label_folder, os.path.join(output_folder, 'labels/test'))

print("Data split into train, validation, and test sets")