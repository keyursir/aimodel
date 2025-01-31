import cv2
import torch
from ultralytics import YOLO

# Load YOLOv5n model
model = YOLO("yolov5n.pt")  # Downloads and loads the pre-trained model

# Initialize webcam
cap = cv2.VideoCapture(1)  # 0 -> Default webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # If no frame is captured, exit

    # Perform object detection
    results = model(frame)

    # Draw bounding boxes on the frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0].item())  # Class index
            
            label = f"{model.names[cls]}: {conf:.2f}"  # Get class name
            
            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2)

    # Display frame
    cv2.imshow("YOLOv5n Object Detection", frame)

    # Exit when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
