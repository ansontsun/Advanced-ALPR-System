import cv2
from ultralytics import YOLOv10
from PIL import Image
import matplotlib.pyplot as plt
import easyocr
import numpy as np

# Initialize YOLOv10 model with weights
#model_path = 'runs/detect/train122/train12/weights/best.pt'
model_path = 'model_weights/best_roboflow.pt'
model = YOLOv10(model_path)

# Path to the input image
image_path = 'dataset/UFPR-ALPR/testing/track0091/track0091[01].png'
#image_path = 'dataset/UFPR-ALPR/testing/track0105/track0105[13].png'
#image_path = 'dataset/UFPR-ALPR/testing/track0145/track0145[21].png'  # Replace with the actual image path
#image_path = 'dataset/UFPR-ALPR/training/track0003/track0003[05].png'
# Run the model on the image
results = model(source=image_path, conf=0.25, save=True)

# Get the bounding box coordinates from the results
if results and len(results[0].boxes) > 0:
    # Extract the first bounding box
    bbox = results[0].boxes[0]

    # Extract coordinates and convert them to integers
    x_min, y_min, x_max, y_max = map(int, bbox.xyxy[0].tolist())

    # Open the image
    image = Image.open(image_path)

    # Crop the image using the bounding box coordinates
    cropped_image = image.crop((x_min-10, y_min-10, x_max+10, y_max+10))
    #cropped_image.save("1.png")
    plt.figure(figsize=(10, 10))
    plt.imshow(cropped_image, cmap='CMRmap')
    plt.axis('off')
    plt.show()

    # Convert PIL image to OpenCV format
    cropped_image_cv = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2BGR)

    # Load pre-trained LapSRN model
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    path = "Scripts/unused/LapSRN_x8.pb"
    sr.readModel(path)
    sr.setModel("lapsrn", 8)

    # Enhance the image
    enhanced_image_cv = sr.upsample(cropped_image_cv)

    # Convert to grayscale
    gray_image = cv2.cvtColor(enhanced_image_cv, cv2.COLOR_BGR2GRAY)

    # # Apply thresholding to clean up the image
    #_, threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convert back to PIL for displaying and OCR
    final_image = Image.fromarray(gray_image)
    #final_image = Image.fromarray(threshold_image)
    # Save the grayscale image to a temporary file
    enhanced_image_path = 'enhanced_image_grayscale.png'
    final_image.save(enhanced_image_path)

    # Display the enhanced image
    plt.figure(figsize=(10, 10))
    plt.imshow(final_image, cmap='gray')
    plt.axis('off')
    plt.show()

    # Initialize EasyOCR reader
    #reader = easyocr.Reader(['en'])  # Use 'en' for English
    reader = easyocr.Reader(['en'], recog_network='best_accuracy')
    #reader = easyocr.Reader(['en'], recog_network='iter_100000')
    # Use EasyOCR to read text from the grayscale image
    ocr_result = reader.readtext(enhanced_image_path)
    #ocr_result = reader.readtext(enhanced_image_path, detail=1, decoder='beamsearch', beamWidth=5, contrast_ths=0.1, adjust_contrast=0.5, text_threshold=0.7, link_threshold=0.4, canvas_size=2560, mag_ratio=1.5)

    # Print the OCR results
    print("OCR Results:")
    for (bbox, text, prob) in ocr_result:
        print(f"Detected text: {text} (Probability: {prob})")
else:
    print("No detections were made.")
