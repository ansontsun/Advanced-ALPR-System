import os
import cv2
import numpy as np
import albumentations as A
import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io

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

class YOLODataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.label_paths = []
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    image_path = os.path.join(subdir, file)
                    self.image_paths.append(image_path)
                    if 'training' in subdir:
                        label_path = image_path.replace('.jpg', '.txt').replace('.png', '.txt')
                    elif 'validation' in subdir:
                        label_path = image_path.replace('.jpg', '_yolo.txt').replace('.png', '_yolo.txt')
                    elif 'testing' in subdir:
                        label_path = image_path.replace('.jpg', '_original.txt').replace('.png', '_original.txt')
                    self.label_paths.append(label_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        image = io.imread(image_path)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        # Apply custom horizontal flip with a probability check
        if np.random.rand() > 0.5:
            image = custom_horizontal_flip(image)
        
        # Load the label
        with open(label_path, 'r') as f:
            labels = f.readlines()
        
        return image, labels

# Define the dataset and data loader
dataset = YOLODataset(root_dir='data/UFPR-ALPR dataset', transform=augmentation_pipeline)
data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Iterate over data loader
for images, labels in data_loader:
    # images is a batch of augmented images
    # labels is a batch of corresponding labels
    # Here you can pass the images and labels to your YOLO model for training
    print(f"Batch of images: {images.shape}")
    print(f"Batch of labels: {labels}")