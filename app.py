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

# List of images to process
image_paths = ['CM YS Jagan launches Disha Patrolling Vehicles _12.jpg']  # Replace with your image paths
results = []

for img_path in image_paths:
    img = cv2.imread(img_path)
    detections = model(img)

    for result in detections:
        for box in result.boxes:
            # Crop the license plate from the image
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_image = img[y1:y2, x1:x2]

            # Recognize the license plate text
            license_plate = recognize_license_plate(plate_image)
            if license_plate:
                results.append(license_plate)

# Save the results to a CSV file
df = pd.DataFrame(results, columns=['License Plate'])
df.to_csv('license_plates.csv', index=False)
