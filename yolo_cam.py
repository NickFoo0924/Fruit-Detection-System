from ultralytics import YOLO
import cv2

# Load YOUR custom trained fruit detection model
model = YOLO("C:/Users/ASUS/OneDrive/Desktop/YOLO_dataset/runs/fruit_detection/weights/best.pt")

print("Model loaded successfully! Press 'q' to quit.")
print(f"Detecting: {model.names}")

# Open your laptop camera (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLO detection on the frame using YOUR custom model
    results = model(frame, conf=0.5)

    # Draw detections on the frame with labels
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow("Fruit Detection (Press 'q' to exit)", annotated_frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()