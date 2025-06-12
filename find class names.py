from ultralytics import YOLO

# Load the model
model = YOLO("best.pt")

# Print class names
print("Number of classes:", len(model.names))
print("Class names:")
for class_id, class_name in model.names.items():
    print(f"{class_id}: {class_name}")
