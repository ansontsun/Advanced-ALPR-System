from ultralytics import YOLOv10
from PIL import Image
import matplotlib.pyplot as plt
import easyocr

# Initialize YOLOv10 model with weights
model_path = 'runs/detect/train122/train12/weights/best.pt'
model = YOLOv10(model_path)

# Path to the input image
image_path = 'dataset/UFPR-ALPR/testing/track0091/track0091[01].png'  # Replace with the actual image path

# Run the model on the image
results = model(source=image_path, conf=0.25, save=True)

# Get the bounding box coordinates from the results
# Assuming results is a list of detection dictionaries
if results and len(results[0].boxes) > 0:
    # Extract the first bounding box
    bbox = results[0].boxes[0]  # This may need adjustment based on the actual structure

    # Extract coordinates and convert them to integers
    x_min, y_min, x_max, y_max = map(int, bbox.xyxy[0].tolist())

    # Print bounding box coordinates
    print(f"Bounding box coordinates: ({x_min}, {y_min}, {x_max}, {y_max})")

    # Open the image
    image = Image.open(image_path)

    # Crop the image using the bounding box coordinates
    cropped_image = image.crop((x_min, y_min, x_max, y_max))

    # Display the cropped image
    plt.figure(figsize=(10, 10))
    plt.imshow(cropped_image)
    plt.axis('off')
    plt.show()

    # Save the cropped image to a file
    cropped_image_path = "cropped_image.png"
    cropped_image.save(cropped_image_path)

    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])  # Use 'en' for English

    # Use EasyOCR to read text from the cropped image
    ocr_result = reader.readtext(cropped_image_path)

    # Print the OCR results
    print("OCR Results:")
    for (bbox, text, prob) in ocr_result:
        print(f"Detected text: {text} (Probability: {prob})")
else:
    print("No detections were made.")
