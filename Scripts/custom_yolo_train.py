import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np

# Define a class name to ID mapping
class_name_to_id = {
    'License_Plate': 0,  # Add other class names and their IDs here
    # Example: 'Car': 1
}

def parse_voc_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in class_name_to_id:
            continue  # Skip unknown class names

        class_id = class_name_to_id[class_name]
        bbox = obj.find('bndbox')
        x_min = int(bbox.find('xmin').text)
        y_min = int(bbox.find('ymin').text)
        x_max = int(bbox.find('xmax').text)
        y_max = int(bbox.find('ymax').text)
        boxes.append([x_min, y_min, x_max, y_max, class_id])
    return boxes

class PascalVOCDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None, max_boxes=50):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.max_boxes = max_boxes
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.jpg')]
        self.annotation_paths = [os.path.join(annotation_dir, fname.replace('.jpg', '.xml')) for fname in
                                 os.listdir(image_dir) if fname.endswith('.jpg')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        annotation_path = self.annotation_paths[idx]

        image = Image.open(image_path).convert("RGB")
        boxes = parse_voc_annotation(annotation_path)

        if len(boxes) == 0:
            boxes = np.zeros((self.max_boxes, 5))  # Create an array with zeros if no boxes are present
        else:
            boxes = np.array(boxes)

        # Pad boxes to max_boxes
        if len(boxes) < self.max_boxes:
            padding = np.zeros((self.max_boxes - len(boxes), 5))  # 5 for [x_min, y_min, x_max, y_max, class_id]
            boxes = np.vstack((boxes, padding))
        elif len(boxes) > self.max_boxes:
            boxes = boxes[:self.max_boxes]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(boxes, dtype=torch.float32)

# Example usage
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for pre-trained models
])

dataset = PascalVOCDataset(
    r'C:\Users\mengbo\PycharmProjects\Advanced-ALPR-System\License Plate Recognition.v4-resized640_aug3x-accurate.voc\test',
    r'C:\Users\mengbo\PycharmProjects\Advanced-ALPR-System\License Plate Recognition.v4-resized640_aug3x-accurate.voc\test',
    transform=transform
)

def collate_fn(batch):
    images, boxes = zip(*batch)
    images = torch.stack(images)
    boxes = torch.stack(boxes)
    return images, boxes

dataloader = DataLoader(
    dataset,
    batch_size=16,  # Try a smaller batch size
    shuffle=True,
    collate_fn=collate_fn
)

# Verify the data loader
for images, boxes in dataloader:
    print(images.shape)  # Should be (batch_size, 3, 224, 224)
    print(boxes.shape)  # Should be (batch_size, num_boxes, 5)
    break

import torch
import torch.nn as nn
import torch.nn.functional as F
num_boxes=3
class LicensePlateDetectionCNN(nn.Module):
    def __init__(self, num_boxes=3):
        super(LicensePlateDetectionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_boxes * 4)  # Predict num_boxes bounding boxes with 4 coordinates each

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, num_boxes, 4)  # Reshape to (batch_size, num_boxes, 4)
        return x

import torch.optim as optim

model = LicensePlateDetectionCNN(num_boxes=3)

criterion = nn.SmoothL1Loss()  # Alternative to MSELoss
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Try a smaller learning rate

num_epochs = 30

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, boxes in dataloader:
        optimizer.zero_grad()
        outputs = model(images)

        # Ensure targets have the same shape as outputs
        valid_boxes = boxes[:, :num_boxes, :4]  # Ensure valid_boxes shape is [batch_size, num_boxes, 4]
        loss = criterion(outputs, valid_boxes)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()


    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}')
    torch.save(model.state_dict(), f'model_weights_epoch_{epoch + 1}.pth')


print('Training complete')

import torch

def calculate_accuracy_without_iou(model, dataloader, threshold=10):
    """
    Calculate the accuracy of the model without using IoU.

    Args:
        model: The trained model.
        dataloader: DataLoader providing the dataset.
        threshold: Allowed margin of error for the bounding box coordinates.

    Returns:
        accuracy: The percentage of correctly predicted bounding boxes.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, boxes in dataloader:
            outputs = model(images)

            # Iterate through each image in the batch
            for i in range(images.size(0)):
                predicted_boxes = outputs[i].cpu().numpy()  # Predicted boxes for the i-th image
                target_boxes = boxes[i].cpu().numpy()  # Target boxes for the i-th image

                for pred_box in predicted_boxes:
                    # Check if predicted box matches any target box within the allowable error margin
                    for target_box in target_boxes:
                        if (
                            abs(pred_box[0] - target_box[0]) <= threshold and
                            abs(pred_box[1] - target_box[1]) <= threshold and
                            abs(pred_box[2] - target_box[2]) <= threshold and
                            abs(pred_box[3] - target_box[3]) <= threshold
                        ):
                            correct += 1
                            break  # If a match is found, stop checking further

                total += len(predicted_boxes)  # Total number of boxes in the current image

    accuracy = correct / total if total > 0 else 0
    return accuracy



# Usage Example
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

dataset = PascalVOCDataset(
    r'C:\Users\mengbo\PycharmProjects\Advanced-ALPR-System\License Plate Recognition.v4-resized640_aug3x-accurate.voc\test',
    r'C:\Users\mengbo\PycharmProjects\Advanced-ALPR-System\License Plate Recognition.v4-resized640_aug3x-accurate.voc\test',
    transform=transform
)

dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn
)

accuracy = calculate_accuracy_without_iou(model, dataloader, threshold=5)
print(f'Accuracy: {accuracy:.4f}')
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
from PIL import Image


def visualize_prediction(model, image_path, transform, threshold=0.1):
    """
    Visualize prediction on a single image.

    Args:
        model: The trained model.        image_path: Path to the image file.
        transform: Transformations to be applied to the image.
        threshold: Allowed margin of error for the bounding box coordinates.
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load and transform the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)

    # Convert tensor to numpy array for visualization
    image_np = image_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
    predicted_boxes = output[0].cpu().numpy()

    # Plot the image with predicted bounding boxes
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(image_np)

    for pred_box in predicted_boxes:
        rect = patches.Rectangle(
            (pred_box[0], pred_box[1]),
            pred_box[2] - pred_box[0],
            pred_box[3] - pred_box[1],
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)

    plt.show()


# Usage Example
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224))
])

image_path = r'C:\Users\mengbo\PycharmProjects\Advanced-ALPR-System\License Plate Recognition.v4-resized640_aug3x-accurate.voc\test\00a7d31c6cc6b7f3_jpg.rf.2707e63f5c51f113de704441ea210a65.jpg'  # Replace with the path to your image
visualize_prediction(model, image_path, transform)
