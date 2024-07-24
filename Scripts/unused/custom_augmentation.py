import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import shutil
import random

def apply_filters(image):
    # Apply random filters for data augmentation
    filters = [
        ("GAUSSIAN_BLUR", ImageFilter.GaussianBlur(2)),
        ("BLUR", ImageFilter.BLUR),
        ("CONTOUR", ImageFilter.CONTOUR),
        ("DETAIL", ImageFilter.DETAIL),
        ("EDGE_ENHANCE", ImageFilter.EDGE_ENHANCE),
        ("SHARPEN", ImageFilter.SHARPEN)
    ]
    filter_name, filter_to_apply = random.choice(filters)
    image = image.filter(filter_to_apply)

    return image, filter_name

def apply_augmentations(image_path):
    # Load image
    image = Image.open(image_path)
    
    # Apply random filters and color augmentations
    filtered_image, filter_name = apply_filters(image)
    
    # Apply random brightness enhancement
    enhancer = ImageEnhance.Brightness(filtered_image)
    brightness = enhancer.enhance(random.uniform(0.5, 1.5))

    # Apply random contrast enhancement
    enhancer = ImageEnhance.Contrast(brightness)
    contrast = enhancer.enhance(random.uniform(0.5, 1.5))

    # Apply random color enhancement
    enhancer = ImageEnhance.Color(contrast)
    saturated = enhancer.enhance(random.uniform(0.5, 1.5))

    return saturated, filter_name

def save_augmented_image_and_label(image_path, label_path, augmented_image, suffix):
    # Save the augmented image
    base, ext = os.path.splitext(image_path)
    augmented_image_path = f"{base}_{suffix}{ext}"
    augmented_image.save(augmented_image_path)
    
    # Duplicate and rename the label file
    new_label_path = f"{base}_{suffix}.txt"
    shutil.copy(label_path, new_label_path)
    
    #print(f"Saved {augmented_image_path} and {new_label_path}")

def process_directory(root_dir):
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".png"):
                image_path = os.path.join(root, file)
                label_path = image_path.replace(".png", ".txt")
                augmented_image, filter_name = apply_augmentations(image_path)
                save_augmented_image_and_label(image_path, label_path, augmented_image, f"augmented_{filter_name}")
        print("Checkpoint")

if __name__ == "__main__":
    root_directory = "../dataset/UFPR-ALPR/training"
    process_directory(root_directory)
