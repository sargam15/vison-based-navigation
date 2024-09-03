import cv2
from ultralytics import YOLO
import time
import os

# Path to your trained YOLOv8 model
model_path = 'C:\\datasets\\coco8\\yolov8n.pt'

# Check if the model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Load the trained YOLOv8 model
model = YOLO(model_path)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set the frame size (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# Directory to save images
save_dir = 'C:\\datasets\\coco8\\runs\\detect'
os.makedirs(save_dir, exist_ok=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Draw bounding boxes and labels on the frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls)]
            conf = box.conf[0]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw label and confidence
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
# Save image of detected object with label
            if x1 < x2 and y1 < y2:  # Check if bounding box coordinates are valid
                obj_image = frame[y1:y2, x1:x2].copy()

                # Add label and confidence to the saved image
                cv2.rectangle(obj_image, (0, 0), (obj_image.shape[1], 20), (0, 255, 0), -1)
                cv2.putText(obj_image, f'{label} {conf:.2f}', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                timestamp = int(time.time() * 1000)  # Use milliseconds for more unique timestamps
                image_path = os.path.join(save_dir, f'detected_object_{label}_{timestamp}.jpg')
                cv2.imwrite(image_path, obj_image)

    # Display the frame
    cv2.imshow('YOLOv8 Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

