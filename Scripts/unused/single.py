import os
import cv2
from ultralytics import YOLOv10
from PIL import Image
import easyocr
import numpy as np

# Initialize YOLOv10 model with weights
model_path = 'model_weights/best.pt'
model = YOLOv10(model_path)

# Specify the path to the single image and corresponding label
image_path = 'dataset/UFPR-ALPR/testing/track0091/track0091[01].png'  # Change this to your image path
label_path = 'dataset/UFPR-ALPR/testing/track0091/track0091[01].txt'  # Change this to your label path

output_file = 'single_image_ocr_results.txt'

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])  # Use 'en' for English

# Load pre-trained LapSRN model
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel("Scripts/unused/LapSRN_x8.pb")  # Path to the LapSRN model
sr.setModel("lapsrn", 8)  # Choose the model and scale factor

with open(output_file, 'w', encoding='utf-8') as f:
    if os.path.exists(label_path):
        # Read the ground truth plate number
        with open(label_path, 'r', encoding='utf-8') as label_file:
            lines = label_file.readlines()
            for line in lines:
                if line.startswith('plate:'):
                    ground_truth_plate = line.split(':')[1].strip().upper().replace(' ', '')

        # Run the model on the image
        results = model(source=image_path, conf=0.25, save=False)

        # Get the bounding box coordinates from the results
        if results and len(results[0].boxes) > 0:
            # Extract the first bounding box
            bbox = results[0].boxes[0]  # This may need adjustment based on the actual structure

            # Extract coordinates and convert them to integers
            x_min, y_min, x_max, y_max = map(int, bbox.xyxy[0].tolist())

            # Open the image
            image = Image.open(image_path)

            # Crop the image using the bounding box coordinates
            cropped_image = image.crop((x_min, y_min, x_max, y_max))

            # Convert PIL image to OpenCV format
            cropped_image_cv = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2BGR)

            # Enhance the image
            enhanced_image_cv = sr.upsample(cropped_image_cv)

            # Convert the enhanced image back to PIL format
            enhanced_image = Image.fromarray(cv2.cvtColor(enhanced_image_cv, cv2.COLOR_BGR2RGB))

            # Save the enhanced image to a temporary file
            enhanced_image_path = 'enhanced_image.png'
            enhanced_image.save(enhanced_image_path)

            # Use EasyOCR to read text from the enhanced image
            ocr_result = reader.readtext(enhanced_image_path)

            # Extract the detected text with the highest probability
            detected_text = ''
            if ocr_result:
                detected_text = ocr_result[0][1].upper().replace(' ', '')

            # Save the results to the output file
            f.write(f'Image: {image_path}\n')
            f.write(f'Ground Truth: {ground_truth_plate}\n')
            f.write(f'OCR Result: {detected_text}\n')
            f.write('\n')
        else:
            f.write(f'Image: {image_path}\n')
            f.write('No detections were made.\n\n')
    else:
        print(f'Label file not found: {label_path}')

print('OCR processing completed. Results saved to:', output_file)
