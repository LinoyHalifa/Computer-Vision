import cv2
import pyttsx3
from ultralytics import YOLO
import numpy as np

# # Loads the lightweight YOLOv8 model
model = YOLO('yolov8n.pt')  # If not found locally, YOLO will download the model automatically


# Text-to-speech engine
engine = pyttsx3.init()

# Function that speaks the given text out loud
def say(text):
    print(f"[AUDIO] {text}")
    engine.say(text)
    engine.runAndWait()

# Open the camera (0 = first camera. If it doesn't work, try 1 or 2)
cap = cv2.VideoCapture(0)

# Check camera connection
if not cap.isOpened():
    print("Camera Is Not Connect")
    exit()
else:
    print("Camera Is Connect")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Object detection using YOLO
    results = model(frame)
    detections = results[0].boxes.data.cpu().numpy()
    annotated_frame = results[0].plot() # The image with bounding boxes around the detected objects


    # Display the image in a new window
    cv2.imshow("BlindVision AR", annotated_frame)

    # Perform detection on only one object per frame
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        cls_name = model.names[int(cls_id)]
        center_x = (x1 + x2) / 2
        width = frame.shape[1]

        # Determine the object's direction within the frame
        if center_x < width / 3:
            direction = "on your left"
        elif center_x > 2 * width / 3:
            direction = "on your right"
        else:
            direction = "in front"

        message = f"{cls_name} {direction}"
        say(message)
        break  # Only one object at a time

    # Exit the loop when pressing the Q key
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:  # esc or q 
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
