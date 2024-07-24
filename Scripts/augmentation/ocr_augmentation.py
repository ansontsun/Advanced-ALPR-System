import os
import cv2
import numpy as np
import albumentations as A
import easyocr
from skimage import io

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Directory containing images and YOLO bounding boxes
root_dir = '/Users/ansonsun/Documents/aps360/project/Advanced-ALPR-System/Scripts/data/UFPR-ALPR dataset'

# Output directory for augmented images
output_dir = 'augmented_data'
os.makedirs(output_dir, exist_ok=True)

# Augmentation pipeline including contrast improvement
augmentation_pipeline = A.Compose([
    A.RandomBrightnessContrast(p=0.2),
    A.CLAHE(clip_limit=2.0, p=0.5),
    A.GaussianBlur(p=0.2),
    A.Rotate(limit=10, p=0.5),
    A.RandomScale(scale_limit=0.2, p=0.3),
])

# Custom function to handle horizontal flip
def custom_horizontal_flip(image, apply_flip=True):
    if apply_flip:
        return cv2.flip(image, 1)  # Horizontal flip
    return image

# Function to extract and augment ROIs
def extract_and_augment_rois(image_path, label_path, augmentations_per_image=5):
    # print(f"Processing {image_path} with labels {label_path}")
    image = io.imread(image_path)
    height, width = image.shape[:2]

    with open(label_path, 'r') as label_file:
        for line in label_file:
            tokens = line.strip().split()
            if len(tokens) < 5:
                print(f"Skipping invalid label line: {line}")
                continue
            class_label = int(tokens[0])
            bbox = [float(token) for token in tokens[1:]]
            x_center, y_center, bbox_width, bbox_height = bbox
            x_center, y_center = x_center * width, y_center * height
            bbox_width, bbox_height = bbox_width * width, bbox_height * height
            x_min = int(x_center - bbox_width / 2)
            x_max = int(x_center + bbox_width / 2)
            y_min = int(y_center - bbox_height / 2)
            y_max = int(y_center + bbox_height / 2)
            roi = image[y_min:y_max, x_min:x_max]

            if roi.size == 0:
                print(f"Skipping empty ROI for image: {image_path}")
                continue

            for i in range(augmentations_per_image):
                # Apply base augmentations
                augmented = augmentation_pipeline(image=roi)
                augmented_image = augmented['image']

                # Apply custom horizontal flip with a probability check
                if np.random.rand() > 0.5:
                    augmented_image = custom_horizontal_flip(augmented_image)

                # Save augmented image
                base_filename = os.path.basename(image_path).replace('.jpg', '').replace('.png', '')
                output_image_path = os.path.join(output_dir, f'{base_filename}_aug_{class_label}_{i}.jpg')
                io.imsave(output_image_path, augmented_image)
                # print(f"Saved augmented image: {output_image_path}")

                # Recognize text using EasyOCR
                # result = reader.readtext(augmented_image)
                # print(f'Results for {output_image_path}:', result)

# Check if root directory exists
if not os.path.exists(root_dir):
    print(f"Root directory does not exist: {root_dir}")
else:
    print(f"Root directory: {root_dir}")

    # List the contents of the root directory
    root_contents = os.listdir(root_dir)
    # print(f"Contents of root directory: {root_contents}")

    # Recursively traverse the directory structure
    for subdir, _, files in os.walk(root_dir):
        print(f"Subdirectory: {subdir}")
        print(f"Files: {files}")

        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                image_path = os.path.join(subdir, file)
                # print(f"Found image file: {image_path}")

                # Determine the appropriate label file suffix
                if 'training' in subdir:
                    label_path = image_path.replace('.jpg', '.txt').replace('.png', '.txt')
                elif 'validation' in subdir:
                    label_path = image_path.replace('.jpg', '_yolo.txt').replace('.png', '_yolo.txt')
                elif 'testing' in subdir:
                    label_path = image_path.replace('.jpg', '_original.txt').replace('.png', '_original.txt')
                else:
                    print(f"Skipping invalid directory: {subdir}")
                    continue

                if os.path.exists(label_path):
                    # print(f"Found label file: {label_path}")
                    extract_and_augment_rois(image_path, label_path)
                else:
                    print(f"Label file does not exist: {label_path}")

print("Data augmentation and preparation for EasyOCR completed successfully.")