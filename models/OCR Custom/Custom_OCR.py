import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import cv2
import os

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation="leaky_relu", skip_conv=True, strides=1, dropout=0.2):
        super(ResidualBlock, self).__init__()
        self.activation = activation
        self.skip_conv = skip_conv
        self.strides = strides
        self.dropout = dropout

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=strides, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if self.skip_conv:
            self.conv_skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides)
            self.bn_skip = nn.BatchNorm2d(out_channels)
        self.dropout_layer = nn.Dropout2d(p=dropout)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.leaky_relu(out, 0.1) if self.activation == "leaky_relu" else F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.skip_conv:
            identity = self.conv_skip(identity)
            identity = self.bn_skip(identity)

        out += identity
        out = F.leaky_relu(out, 0.1) if self.activation == "leaky_relu" else F.relu(out)

        if self.dropout:
            out = self.dropout_layer(out)

        return out


class OCRModel(nn.Module):
    def __init__(self, input_dim, output_dim, activation="leaky_relu", dropout=0.2):
        super(OCRModel, self).__init__()
        self.input_dim = input_dim
        self.activation = activation
        self.dropout = dropout

        self.residual_block1 = ResidualBlock(input_dim[0], 32, activation=activation, skip_conv=True, strides=1,
                                             dropout=dropout)
        self.residual_block2 = ResidualBlock(32, 32, activation=activation, skip_conv=True, strides=2, dropout=dropout)
        self.residual_block3 = ResidualBlock(32, 32, activation=activation, skip_conv=False, strides=1, dropout=dropout)
        self.residual_block4 = ResidualBlock(32, 64, activation=activation, skip_conv=True, strides=2, dropout=dropout)
        self.residual_block5 = ResidualBlock(64, 64, activation=activation, skip_conv=False, strides=1, dropout=dropout)
        self.residual_block6 = ResidualBlock(64, 128, activation=activation, skip_conv=True, strides=1, dropout=dropout)
        self.residual_block7 = ResidualBlock(128, 128, activation=activation, skip_conv=False, strides=1,
                                             dropout=dropout)

        self.lstm = nn.LSTM(128 * (input_dim[1] // 4) * (input_dim[2] // 4), 128, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256, output_dim + 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = x / 255.0

        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = self.residual_block3(x)
        x = self.residual_block4(x)
        x = self.residual_block5(x)
        x = self.residual_block6(x)
        x = self.residual_block7(x)

        b, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)

        x, _ = self.lstm(x)
        x = self.fc(x)
        x = self.softmax(x)

        return x




class LicensePlateDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        with open(label_file, 'r') as f:
            lines = f.readlines()
            self.labels = [line.strip().split(',') for line in lines]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name, label = self.labels[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if self.transform:
            image = self.transform(image)
        label = np.array([int(ch) for ch in label])  # Adjust as per the label format
        return image, label


def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    lengths = [len(label) for label in labels]
    labels = torch.cat([torch.tensor(label) for label in labels])
    return images, labels, lengths


# Adjust paths to your dataset
img_dir = 'path_to_images'
label_file = 'path_to_labels.csv'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 256))  # Adjust size based on your model input
])

dataset = LicensePlateDataset(img_dir, label_file, transform=transform)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

input_dim = (1, 128, 256)
output_dim = 36  # Adjust based on the number of characters in license plates (0-9, A-Z)

model = OCRModel(input_dim, output_dim)
criterion = nn.CTCLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(10):  # Number of epochs
    for batch_idx, (data, target, lengths) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)

        # Prepare input for CTC Loss
        input_lengths = torch.full(size=(output.size(1),), fill_value=output.size(0), dtype=torch.long)
        target_lengths = torch.tensor(lengths, dtype=torch.long)

        loss = criterion(output, target, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f'Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item()}')


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((128, 256)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("L")
    image = transform(image)
    image = image.unsqueeze(0)  # Add a batch dimension
    return image


def decode_output(output):
    output = output.squeeze(0)  # Remove the batch dimension
    output = torch.argmax(output, dim=-1)  # Get the index of the highest probability
    output = output.numpy()  # Convert to numpy array
    return output


def detect_license_plate(image_path, model_path='ocr_model.pth'):
    model = OCRModel(input_dim=(1, 128, 256), output_dim=36)  # Adjust input_dim and output_dim
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    input_image = preprocess_image(image_path)

    with torch.no_grad():  # No need to calculate gradients for inference
        output = model(input_image)

    decoded_output = decode_output(output)
    print('Predicted sequence:', decoded_output)


# Example usage
image_path = 'path_to_your_license_plate_image.png'
detect_license_plate(image_path)
