from ultralytics import YOLO
import cv2
import pandas as pd
import pytesseract

# Set the path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust this if necessary

# Load the YOLO model
model = YOLO('number_plate_model_best.pt')

# Define the function for license plate recognition
def recognize_license_plate(plate_image):
    gray_plate = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    license_text = pytesseract.image_to_string(gray_plate, config='--psm 8')
    return license_text.strip()

# Open video file or capture from webcam (0 for default webcam)
video_source = 'car.mp4'  # Change this to 0 for webcam
cap = cv2.VideoCapture(video_source)

# List to store recognized license plates
results = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform prediction
    results_model = model.predict(frame)

    # Draw bounding boxes and class labels on the frame based on the results
    for result in results_model:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get the coordinates of the bounding box
            class_id = int(box.cls[0])  # Get the class ID
            confidence = box.conf[0]     # Get the confidence score

            # Check if the detected object is a license plate
            if class_id == 0:  # Assuming class ID 0 is for license plates
                # Crop the license plate from the image
                plate_image = frame[y1:y2, x1:x2]
                
                # Recognize the license plate text
                license_plate = recognize_license_plate(plate_image)
                if license_plate:
                    results.append(license_plate)  # Store the recognized license plate

                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Put the class label and recognized license plate above the bounding box
                label = f"License Plate: {license_plate} {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the frame with predictions
    cv2.imshow('Predicted Video', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()

# Save the results to a CSV file
df = pd.DataFrame(results, columns=['License Plate'])
df.to_csv('license_plates.csv', index=False)
