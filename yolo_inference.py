from ultralytics import YOLO

# Load the model
model = YOLO('models/best.pt')

# Perform inference
results = model.predict('input_videos/08fd33_4.mp4',save=True)
print(results[0])
print("===========================================")

# Print the bounding boxes
for box in results[0].boxes:
    print(box)
