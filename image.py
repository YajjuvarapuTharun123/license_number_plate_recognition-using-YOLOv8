from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO('number_plate_model_best.pt')

# Perform prediction
results = model.predict('CM YS Jagan launches Disha Patrolling Vehicles _12.jpg')

# Load the original image
image = cv2.imread('CM YS Jagan launches Disha Patrolling Vehicles _12.jpg')

# Draw bounding boxes and class labels on the image based on the results
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0]  # Get the coordinates of the bounding box
        class_id = int(box.cls[0])  # Get the class ID
        confidence = box.conf[0]     # Get the confidence score

        # Get class name (you can customize this mapping based on your model's classes)
        class_name = model.names[class_id] if model.names else f"Class {class_id}"

        # Draw the bounding box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Put the class label and confidence score above the bounding box
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Display the image with predictions
cv2.imshow('Predicted Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
