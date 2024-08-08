import cv2
from ultralytics import YOLOv10  # Assuming this is hypothetical or custom; adjust according to the actual import
from PIL import Image
import matplotlib.pyplot as plt
import pytesseract
import numpy as np

# Initialize YOLOv10 model with weights
model_path = 'runs/detect/train122/train12/weights/best.pt'
model = YOLOv10(model_path)  # Adjust based on actual usage

# Path to the input image
image_path = 'dataset/UFPR-ALPR/testing/track0091/track0091[13].png'  # Replace with the actual image path

# Run the model on the image
results = model(source=image_path, conf=0.25, save=True)

# Get the bounding box coordinates from the results
if results and len(results[0].boxes) > 0:
    bbox = results[0].boxes[0]

    # Extract coordinates and convert them to integers
    x_min, y_min, x_max, y_max = map(int, bbox.xyxy[0].tolist())

    # Open the image
    image = Image.open(image_path)

    # Crop the image using the bounding box coordinates, adjusting margins if needed
    cropped_image = image.crop((x_min-10, y_min-10, x_max+10, y_max+10))

    # Convert PIL image to OpenCV format
    cropped_image_cv = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2BGR)

    # Load pre-trained LapSRN model to enhance image resolution
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    path = "Scripts/unused/LapSRN_x8.pb"
    sr.readModel(path)
    sr.setModel("lapsrn", 8)

    # Enhance the image
    enhanced_image_cv = sr.upsample(cropped_image_cv)

    # Convert to grayscale
    gray_image = cv2.cvtColor(enhanced_image_cv, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to clean up the image
    _, threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convert back to PIL for pytesseract
    final_image = Image.fromarray(threshold_image)

    # Save the final image for inspection (optional)
    final_image_path = 'final_image_for_ocr.png'
    final_image.save(final_image_path)

    # Display the final processed image
    plt.figure(figsize=(10, 10))
    plt.imshow(final_image, cmap='gray')
    plt.axis('off')
    plt.show()

    # Use pytesseract to read text from the final processed image
    pytesseract.pytesseract.tesseract_cmd = r'C:/Users/69edw/AppData/Local/Programs/Tesseract-OCR/tesseract.exe'
    ocr_result = pytesseract.image_to_string(final_image)

    # Print the OCR results
    print("OCR Results:")
    print(ocr_result)
else:
    print("No detections were made.")
