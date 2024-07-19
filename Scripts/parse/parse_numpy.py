import os
import cv2
import numpy as np
from skimage.feature import hog

# Root directory containing nested folders of images and labels
root_dir = 'data/UFPR-ALPR dataset'

# Image preprocessing function
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    return image

# Function to extract bounding box coordinates
def extract_bounding_box(image, bbox):
    height, width = image.shape
    x_center, y_center, bbox_width, bbox_height = bbox
    x_center, y_center = x_center * width, y_center * height
    bbox_width, bbox_height = bbox_width * width, bbox_height * height
    x_min = int(x_center - bbox_width / 2)
    x_max = int(x_center + bbox_width / 2)
    y_min = int(y_center - bbox_height / 2)
    y_max = int(y_center + bbox_height / 2)
    roi = image[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return None
    return cv2.resize(roi, (640, 640))  # Resize to a fixed size

# HOG feature extraction
def extract_hog_features(image):
    features, _ = hog(image, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
    return features

# Generate background ROIs
def generate_background_rois(image, num_rois=5, roi_size=(640, 640)):
    height, width = image.shape
    rois = []
    for _ in range(num_rois):
        x_min = np.random.randint(0, width - roi_size[1])
        y_min = np.random.randint(0, height - roi_size[0])
        x_max = x_min + roi_size[1]
        y_max = y_min + roi_size[0]
        rois.append(image[y_min:y_max, x_min:x_max])
    return rois

# Initialize lists to hold features and labels
features = []
processed_labels = []

# Recursively traverse the directory structure
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):  # Assuming images are in jpg or png format
            image_path = os.path.join(subdir, file)
            
            # Determine the appropriate label file suffix
            if 'training' in subdir:
                label_path = image_path.replace('.jpg', '.txt').replace('.png', '.txt')
            elif 'validation' in subdir:
                label_path = image_path.replace('.jpg', '_yolo.txt').replace('.png', '_yolo.txt')
            elif 'testing' in subdir:
                label_path = image_path.replace('.jpg', '_original.txt').replace('.png', '_original.txt')
            else:
                print(f"Skipping invalid directory: {subdir}")
                continue  # Skip directories that don't match the expected structure

            # Preprocess the image and read the label
            try:
                image = preprocess_image(image_path)
                if image is None:
                    print(f"Error loading image: {image_path}")
                    continue
                
                with open(label_path, 'r') as label_file:
                    for line in label_file:
                        tokens = line.strip().split()
                        if len(tokens) < 5:
                            print(f"Invalid label format in file {label_path}: {line}")
                            continue  # Skip lines that don't have enough tokens
                        class_label = int(tokens[0])
                        bbox = [float(token) for token in tokens[1:]]
                        roi = extract_bounding_box(image, bbox)
                        if roi is None:
                            print(f"Skipping empty ROI for image: {image_path}")
                            continue  # Skip empty ROIs
                        hog_features = extract_hog_features(roi)
                        features.append(hog_features)
                        processed_labels.append(1)  # 1 for license plate (positive class)
                        print(f"Extracted features from ROI for image: {image_path}")

                # Generate and append background ROIs
                background_rois = generate_background_rois(image)
                for roi in background_rois:
                    hog_features = extract_hog_features(cv2.resize(roi, (640, 640)))
                    features.append(hog_features)
                    processed_labels.append(0)  # 0 for background (negative class)

            except Exception as e:
                print(f"Error processing {image_path} or {label_path}: {e}")

# Check if features and labels are populated
if len(features) == 0 or len(processed_labels) == 0:
    print("No features or labels were extracted. Please check the data and paths.")
else:
    # Convert to numpy arrays
    features = np.array(features)
    processed_labels = np.array(processed_labels)

    # Save the features and labels to .npy files
    np.save('features.npy', features)
    np.save('labels.npy', processed_labels)

    print("Features and labels saved successfully.")
    
    