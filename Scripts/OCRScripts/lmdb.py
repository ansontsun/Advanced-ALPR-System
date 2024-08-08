import os
from ultralytics import YOLOv10
from PIL import Image

# Initialize YOLOv10 model with weights
model_path = 'model_weights/best.pt'
model = YOLOv10(model_path)

# Path to the test directory
test_dir = 'dataset/UFPR-ALPR/testing'
output_dir = 'test_cropped_images'
output_file = 'test_truth.txt'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read OCR results from the existing file
ocr_results_file = 'train_ocr_results.txt'
ocr_results = {}

with open(ocr_results_file, 'r', encoding='utf-8') as file:
    lines = file.readlines()
    current_image = None
    for line in lines:
        if line.startswith('Image:'):
            current_image = line.split('Image: ')[1].strip()
        elif line.startswith('Ground Truth:'):
            ground_truth = line.split('Ground Truth: ')[1].strip().upper()
        elif line.startswith('OCR Result:'):
            ocr_result = line.split('OCR Result: ')[1].strip().upper()
            ocr_results[current_image] = (ground_truth, ocr_result)

# Function to clean OCR result (based on previous script)
def clean_ocr_result(ocr_result):
    dict_char_to_int = {
        'O': '0',
        'I': '1',
        'J': '3',
        'A': '4',
        'G': '6',
        'S': '5',
        '|': '1',
        '\\': '1'
    }

    dict_int_to_char = {
        '0': 'O',
        '1': 'I',
        '3': 'J',
        '4': 'A',
        '6': 'G',
        '5': 'S',
        '1': '|',
        '1': '\\'
    }

    import re
    cleaned_result = re.sub(r'[^a-zA-Z0-9|\\]', '', ocr_result).upper()

    if len(cleaned_result) != 7:
        return cleaned_result

    cleaned_result_list = list(cleaned_result)
    for i in range(3):
        if cleaned_result_list[i] in dict_int_to_char:
            cleaned_result_list[i] = dict_int_to_char[cleaned_result_list[i]]

    for i in range(3, 7):
        if cleaned_result_list[i] in dict_char_to_int:
            cleaned_result_list[i] = dict_char_to_int[cleaned_result_list[i]]

    return ''.join(cleaned_result_list)

# Process the images
with open(output_file, 'w', encoding='utf-8') as f:
    for image_path, (ground_truth, ocr_result) in ocr_results.items():
        cleaned_ocr_result = clean_ocr_result(ocr_result)

        if cleaned_ocr_result == ground_truth:
            # Run the model on the image
            results = model(source=image_path, conf=0.25, save=False)

            # Get the bounding box coordinates from the results
            if results and len(results[0].boxes) > 0:
                # Extract the first bounding box
                bbox = results[0].boxes[0]

                # Extract coordinates and convert them to integers
                x_min, y_min, x_max, y_max = map(int, bbox.xyxy[0].tolist())

                # Open the image
                image = Image.open(image_path)

                # Crop the image using the bounding box coordinates
                cropped_image = image.crop((x_min, y_min, x_max, y_max))

                # Save the cropped image to the output directory
                cropped_image_name = os.path.splitext(os.path.basename(image_path))[0] + '_cropped.png'
                cropped_image_path = os.path.join(output_dir, cropped_image_name)
                cropped_image.save(cropped_image_path)

                # Write the cropped image path and ground truth plate to the output file
                f.write(f'{cropped_image_path} {ground_truth}\n')

print('Processing completed. Cropped images and ground truth saved to:', output_file)
