import os
import torch
from torch.utils.data import DataLoader, Dataset
import cv2
import glob
import yaml
from pathlib import Path
from ultralytics import YOLO

#We are currently using the train script from yolov5 directly: 
# python train.py --img 640 --batch 16 --epochs 100 --data ../data.yaml --cfg models/yolov5s.yaml --weights yolov5s.pt --device 0

# Custom Dataset Class
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

# Load Data Configuration
def load_data_config(data_config_path):
    with open(data_config_path, 'r') as file:
        data_config = yaml.safe_load(file)
    return data_config

# Train the Model
def train_yolov8(data_config_path, model_cfg, weights_path, img_size, batch_size, epochs, device):
    data_config = load_data_config(data_config_path)
    train_dataset = CustomYoloDataset(img_dir=data_config['train'])
    val_dataset = CustomYoloDataset(img_dir=data_config['val'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = YOLO(model_cfg)
    model.load(weights_path)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            images, targets = data['image'].to(device), data['boxes'].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch [{epoch}/{epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Validation loop (if necessary)
        model.eval()
        with torch.no_grad():
            for data in val_loader:
                images, targets = data['image'].to(device), data['boxes'].to(device)
                outputs = model(images)
                # Compute validation metrics if needed

    # Save the trained model
    model_path = f"yolov8_trained_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    data_config_path = "data.yaml"
    model_cfg = "models/yolov8s.yaml"  # Adjust path if necessary
    weights_path = "yolov8s.pt"  # Path to pretrained weights
    img_size = 640
    batch_size = 16
    epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_yolov8(data_config_path, model_cfg, weights_path, img_size, batch_size, epochs, device)
