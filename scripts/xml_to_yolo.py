import os
import xml.etree.ElementTree as ET

# Paths
xml_folder = "F:/Real-Time-Elephant-Detection/data/xml_labels"
output_folder = "F:/Real-Time-Elephant-Detection/data/yolo_labels"
os.makedirs(output_folder, exist_ok=True)

# Class map
class_mapping = {"elephant": 0}

# Conversion function
def convert_voc_to_yolo(xml_file, output_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Get image dimensions
    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)

    print(f"Processing {xml_file} - Width: {width}, Height: {height}")

    yolo_annotations = []

    for obj in root.findall("object"):
        class_name = obj.find("name").text.strip()  # Extract the text and remove whitespace
        print(f"Found class: {class_name}")

        if class_name not in class_mapping:
            print(f"Skipping class: {class_name}")
            continue
        class_id = class_mapping[class_name]

        # Get bounding box coordinates
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        print(f"Bounding Box: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")

        # Convert to YOLO format
        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        bbox_width = (xmax - xmin) / width
        bbox_height = (ymax - ymin) / height

        # Append to YOLO annotations
        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")

    # Write to YOLO file
    if not yolo_annotations:
        print(f"No annotations found for {xml_file}")

    with open(output_file, "w") as f:
        f.write("\n".join(yolo_annotations))
    
# Process all XML files
for xml_file in os.listdir(xml_folder):
    if xml_file.endswith(".xml"):
        input_path = os.path.join(xml_folder, xml_file)
        output_path = os.path.join(output_folder, xml_file.replace(".xml", ".txt"))
        convert_voc_to_yolo(input_path, output_path)

print(f"Conversion complete! YOLO labels saved in {output_folder}")