import os
import cv2
from ultralytics import YOLOv10
from PIL import Image
import easyocr
import numpy as np

# Initialize YOLOv10 model with weights
model_path = 'runs/detect/train122/train12/weights/best.pt'
model = YOLOv10(model_path)

# Path to the main test directory containing multiple tracks
main_test_dir = 'dataset/UFPR-ALPR/testing'
output_file = 'track_best_ocr_results.txt'

# Initialize EasyOCR reader with specified parameters
reader = easyocr.Reader(['en'])  # Use 'en' for English

# Load pre-trained LapSRN model
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel("Scripts/unused/LapSRN_x8.pb")
sr.setModel("lapsrn", 8)

# Open the output file for writing results
with open(output_file, 'w', encoding='utf-8') as f:
    # Traverse through each track subdirectory
    for track_dir in os.listdir(main_test_dir):
        track_path = os.path.join(main_test_dir, track_dir)
        if os.path.isdir(track_path):
            best_ocr_text = ""
            best_ocr_prob = 0
            best_image_path = ""
            ground_truth_plate = ""

            # Process each image in the track directory
            for file in os.listdir(track_path):
                if file.endswith('.png'):
                    image_path = os.path.join(track_path, file)
                    label_path = image_path.replace('.png', '_original.txt')

                    # Optional: Retrieve ground truth label if available
                    if os.path.exists(label_path):
                        with open(label_path, 'r', encoding='utf-8') as label_file:
                            lines = label_file.read().strip()
                            if lines.startswith('plate:'):
                                ground_truth_plate = lines.split(':')[1].strip()

                    # Run the model on the image
                    results = model(source=image_path, conf=0.25, save=False)

                    # Get the bounding box coordinates from the results
                    if results and len(results[0].boxes) > 0:
                        bbox = results[0].boxes[0]
                        x_min, y_min, x_max, y_max = map(int, bbox.xyxy[0].tolist())
                        image = Image.open(image_path)
                        cropped_image = image.crop((x_min-10, y_min-10, x_max+10, y_max+10))
                        cropped_image_cv = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2BGR)
                        enhanced_image_cv = sr.upsample(cropped_image_cv)
                        gray_image = cv2.cvtColor(enhanced_image_cv, cv2.COLOR_BGR2GRAY)
                        final_image = Image.fromarray(gray_image)
                        enhanced_image_path = 'enhanced_image_grayscale.png'
                        final_image.save(enhanced_image_path)
                        ocr_results = reader.readtext(enhanced_image_path, detail=1, decoder='beamsearch', beamWidth=5, contrast_ths=0.1, adjust_contrast=0.5, text_threshold=0.7, link_threshold=0.4, canvas_size=2560, mag_ratio=1.5)

                        # Evaluate OCR results
                        for result in ocr_results:
                            text = result[1].upper().replace(' ', '').replace('|', '1').replace('/', '1')
                            if 6 <= len(text) <= 7 and result[2] > best_ocr_prob:
                                best_ocr_prob = result[2]
                                best_ocr_text = text
                                best_image_path = image_path

            # Write the best result for this track to the file
            f.write(f'Track: {track_dir}\n')
            f.write(f'Image: {best_image_path}\n')
            f.write(f'Ground Truth: {ground_truth_plate}\n')
            f.write(f'OCR Result: {best_ocr_text} (Probability: {best_ocr_prob})\n\n')

print('OCR processing for all tracks completed. Results saved to:', output_file)
