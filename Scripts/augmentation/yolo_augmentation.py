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
output_dir = 'augmented_data_yolo'
os.makedirs(output_dir, exist_ok=True)

# Augmentation pipeline including contrast improvement and handling bounding boxes
augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.CLAHE(clip_limit=2.0, p=0.5),
    A.OneOf([
            A.GaussNoise(p=0.1),
            A.GaussianBlur(p=0.1),
    ], p=0.2),
    A.Rotate(limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT),
    A.RandomScale(scale_limit=0.2, p=0.3)
], bbox_params=A.BboxParams(format='yolo', label_fields=['labels']))

# Function to augment images and process labels
def augment_and_process_image(image_path, augmentations_per_image=5):
    print(f"Processing {image_path}")
    image = io.imread(image_path)
    height, width = image.shape[:2]

    # Load YOLO labels
    label_path = image_path.replace('.jpg', '.txt').replace('.png', '.txt')
    bboxes = []
    labels = []

    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.strip().split()
            if len(tokens) < 5:
                print(f"Skipping invalid label line: {line}")
                continue
            class_label = int(tokens[0])
            bbox = [float(token) for token in tokens[1:]]
            bboxes.append(bbox)
            labels.append(class_label)
    
    for i in range(augmentations_per_image):
        augmented = augmentation_pipeline(image=image, bboxes=bboxes, labels=labels)
        augmented_image = augmented['image']
        augmented_bboxes = augmented['bboxes']

        # Save augmented image
        base_filename = os.path.basename(image_path).replace('.jpg', '').replace('.png', '')
        output_image_path = os.path.join(output_dir, f'{base_filename}_aug_{i}.jpg')
        io.imsave(output_image_path, augmented_image)
        # print(f"Saved augmented image: {output_image_path}")

        # Save augmented labels
        output_label_path = os.path.join(output_dir, f'{base_filename}_aug_{i}.txt')
        with open(output_label_path, 'w') as f:
            for bbox, label in zip(augmented_bboxes, labels):
                f.write(f"{label} {' '.join(map(str, bbox))}\n")
        # print(f"Saved augmented labels: {output_label_path}")

# Check if root directory exists
if not os.path.exists(root_dir):
    print(f"Root directory does not exist: {root_dir}")
else:
    # print(f"Root directory: {root_dir}")

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
                    augment_and_process_image(image_path)
                else:
                    print(f"Label file does not exist: {label_path}")

print("Data augmentation and preparation for YOLO completed successfully.")