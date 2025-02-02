from yolo_training import model

# Load the trained YOLOv8 model
results = model.val(data='data.yaml', split='test')
print(results)