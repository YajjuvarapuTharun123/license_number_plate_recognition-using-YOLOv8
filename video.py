from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO('number_plate_model_best.pt')

# Open video file or capture from webcam (0 for default webcam)
video_source = 'car.mp4'  # Change this to 0 for webcam
cap = cv2.VideoCapture(video_source)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform prediction
    results = model.predict(frame)

    # Draw bounding boxes and class labels on the frame based on the results
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Get the coordinates of the bounding box
            class_id = int(box.cls[0])  # Get the class ID
            confidence = box.conf[0]     # Get the confidence score

            # Get class name
            class_name = model.names[class_id] if model.names else f"Class {class_id}"

            # Draw the bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Put the class label and confidence score above the bounding box
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the frame with predictions
    cv2.imshow('Predicted Video', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
