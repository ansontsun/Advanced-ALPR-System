import os
import cv2
import numpy as np

# Root directory containing nested folders of images and labels
root_dir = 'data/UFPR-ALPR dataset'

# Image preprocessing function
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    image = cv2.resize(image, (28, 28))  # Resize to a fixed size (e.g., 28x28)
    image = image.flatten()  # Flatten the image to a 1D array
    return image

# Initialize lists to hold features and labels
features = []
processed_labels = []

# Function to check if a string is a number
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

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
                continue  # Skip directories that don't match the expected structure

            # Preprocess the image and read the label
            try:
                features.append(preprocess_image(image_path))
                
                with open(label_path, 'r') as label_file:
                    for line in label_file:
                        first_token = line.strip().split()[0]
                        if is_number(first_token):  # Only process lines with numeric labels
                            label = first_token
                            processed_labels.append(int(label))  # Convert label to integer
                            break  # Stop after the first valid label

            except Exception as e:
                print(f"Error processing {image_path} or {label_path}: {e}")

# Convert lists to numpy arrays
features = np.array(features)
processed_labels = np.array(processed_labels)

# Save the features and labels to .npy files
np.save('features.npy', features)
np.save('labels.npy', processed_labels)

print("Features and labels saved successfully.")

