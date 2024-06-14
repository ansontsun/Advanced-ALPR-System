import os
import glob
import cv2
import torch
from torch.utils.data import Dataset

class CustomYoloDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_files = sorted(glob.glob(os.path.join(img_dir, '*.png')))
        self.label_files = sorted(glob.glob(os.path.join(img_dir, '*.txt')))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        label_path = self.label_files[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        boxes = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                cls, x_center, y_center, width, height = map(float, line.strip().split())
                boxes.append([cls, x_center, y_center, width, height])
        
        sample = {'image': image, 'boxes': boxes}

        if self.transform:
            sample = self.transform(sample)

        return sample
